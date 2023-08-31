from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import time
import os
# from six.moves import cPickle
import pickle as cPickle

import logging
import sys
import opts
import models
from dataloader import *
import eval_utils
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward

try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def setup_logging(opt):
    logger = logging.getLogger('log')
    logger.setLevel(logging.INFO)
    
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # if not os.path.exists(cfg.ROOT_DIR):
    #     os.makedirs(cfg.ROOT_DIR)
    
    fh = logging.FileHandler(os.path.join(opt.checkpoint_path, 'log.txt'))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # logger.info('Training with config:')
    # logger.info(pprint.pformat(cfg))
    return logger

def train(opt, logger):
    # Deal with feature things before anything
    opt.use_att = utils.if_use_att(opt.caption_model)
    if opt.use_box: opt.att_feat_size = opt.att_feat_size + 5

    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)

    infos = {}
    histories = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+'-best.pkl'), 'rb') as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

        # if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
        #     with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')) as f:
        #         histories = cPickle.load(f)

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)

    # val_result_history = histories.get('val_result_history', {})
    # loss_history = histories.get('loss_history', {})
    # lr_history = histories.get('lr_history', {})
    # ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)
    

    model = models.setup(opt).cuda()
    dp_model = torch.nn.DataParallel(model)

    # for p in model.parameters():
    #     print (p)
    #     print (p.is_cuda)
    #     time.sleep(3)

    epoch_done = True
    # Assure in training mode
    dp_model.train()

    if opt.label_smoothing > 0:
        crit = utils.LabelSmoothing(smoothing=opt.label_smoothing)
    else:
        crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()

    if opt.noamopt:
        assert opt.caption_model == 'transformer', 'noamopt can only work with transformer'
        optimizer = utils.get_std_opt(model, factor=opt.noamopt_factor, warmup=opt.noamopt_warmup)
        optimizer._step = iteration
    elif opt.reduce_on_plateau:
        optimizer = utils.build_optimizer(model.parameters(), opt)
        optimizer = utils.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    else:
        optimizer = utils.build_optimizer(model.parameters(), opt)
    # Load the optimizer
    if vars(opt).get('start_from', None) is not None and os.path.isfile(os.path.join(opt.start_from,"optimizer-best.pth")):
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer-best.pth')))

    def save_checkpoint(model, infos, optimizer, histories=None, append=''):
        if len(append) > 0:
            append = '-' + append
        # if checkpoint_path doesn't exist
        if not os.path.isdir(opt.checkpoint_path):
            os.makedirs(opt.checkpoint_path)
        checkpoint_path = os.path.join(opt.checkpoint_path, 'model%s.pth' %(append))
        torch.save(model.state_dict(), checkpoint_path)
        print("model saved to {}".format(checkpoint_path))
        if optimizer:
            optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer%s.pth' %(append))
            torch.save(optimizer.state_dict(), optimizer_path)
        with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'%s.pkl' %(append)), 'wb') as f:
            cPickle.dump(infos, f)
        if histories:
            with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+'%s.pkl' %(append)), 'wb') as f:
                cPickle.dump(histories, f)

    def evaluate(dp_model, model, epoch, loader, iteration, optimizer, infos, tb_summary_writer, best_val_score):
        eval_kwargs = {'split': 'val',
                        'dataset': opt.input_json,
                        'verbose': False,
                        'result_dir': opt.result_dir}
        eval_kwargs.update(vars(opt))
        print('Evaluating ...')
        if opt.beam_size == 1:
            val_loss, predictions, lang_stats = eval_utils.eval_split(dp_model, crit, loader, eval_kwargs)
        else:
            val_loss, predictions, lang_stats = eval_utils.eval_split(model, crit, loader, eval_kwargs)

        with open(os.path.join(opt.result_dir, 'eval_results/' + opt.id + '_' + 'val' + '.txt'), 'a+') as out:
            out.write('\nEpoch %s iter %s:'%(epoch,iteration) + '\n')
            for metric in ['CIDEr', 'METEOR', ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"], 'ROUGE_L']:
                if type(metric) == list:
                    for sub_metric in ["Bleu_4", "Bleu_3", "Bleu_2", "Bleu_1"]:
                        score = lang_stats[sub_metric]
                        out.write('%s: %.4f'%(sub_metric, score) + '\n')
                else:
                    score = lang_stats[metric]
                    out.write('%s: %.4f'%(metric, score) + '\n')

        # logger
        logger.info('######## Epoch ' + str(epoch) + ' ########')
        logger.info(str(lang_stats))

        if opt.reduce_on_plateau:
            if 'CIDEr' in lang_stats:
                optimizer.scheduler_step(-lang_stats['CIDEr'])
            else:
                optimizer.scheduler_step(val_loss)

        # Write validation result into summary
        add_summary_value(tb_summary_writer, 'validation loss', val_loss, iteration)
        for k,v in lang_stats.items():
            add_summary_value(tb_summary_writer, k, v, iteration)

        # Save model if is improving on validation result
        if opt.language_eval == 1:
            current_score = lang_stats['CIDEr']
        else:
            current_score = - val_loss

        best_flag = False
        if best_val_score is None or current_score > best_val_score:
            best_val_score = current_score
            best_flag = True

        # Dump miscalleous informations
        infos['iter'] = iteration
        infos['epoch'] = epoch
        infos['iterators'] = loader.iterators
        infos['split_ix'] = loader.split_ix
        infos['best_val_score'] = best_val_score
        infos['opt'] = opt
        infos['vocab'] = loader.get_vocab()
        save_checkpoint(model, infos, optimizer, append=str(iteration))

        if best_flag:
            save_checkpoint(model, infos, optimizer, append='best')

        return val_loss, predictions, lang_stats, optimizer, infos, tb_summary_writer, best_val_score

    # start = time.time()
    data_time = utils.AverageMeter()
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    while True:
        if epoch_done:
            if not opt.noamopt and not opt.reduce_on_plateau:
                # Assign the learning rate
                if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                    frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                    decay_factor = opt.learning_rate_decay_rate  ** frac
                    opt.current_lr = opt.learning_rate * decay_factor
                else:
                    opt.current_lr = opt.learning_rate
                utils.set_lr(optimizer, opt.current_lr) # set the decayed rate
            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob

            # If start self critical training
            if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                sc_flag = True
                init_scorer(opt.cached_tokens)
            else:
                sc_flag = False

            epoch_done = False
                
        start = time.time()
        # Load data from train split (0)
        data = loader.get_batch('train')
        # if iteration % (opt.print_every*5) == 0:
        #     print('Read data:', time.time() - start)
        data_time.update(time.time() - start)

        torch.cuda.synchronize()
        # start = time.time()

        tmp = [data['ocr_feats'], data['ocr_masks'], data['ocr_relations'], data['fc_feats'], data['att_feats'], data['labels'], data['target_labels'], data['masks'], data['att_masks']]
        tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
        ocr_feats, ocr_masks, ocr_relations, fc_feats, att_feats, labels, target_labels, masks, att_masks = tmp
        
        optimizer.zero_grad()
        if not sc_flag:
            loss = crit(dp_model(ocr_feats, ocr_masks, ocr_relations, fc_feats, att_feats, labels, att_masks), target_labels, masks[:,1:])
        else:
            gen_result, sample_logprobs = dp_model(fc_feats, att_feats, att_masks, opt={'sample_max':0}, mode='sample')
            reward = get_self_critical_reward(dp_model, fc_feats, att_feats, att_masks, data, gen_result, opt)
            loss = rl_crit(sample_logprobs, gen_result.data, torch.from_numpy(reward).float().cuda())

        # # used for testing the best model
        # evaluate(dp_model, model, epoch, loader, iteration, optimizer, infos, tb_summary_writer, best_val_score)
        # input()

        loss.backward()
        # utils.clip_gradient(optimizer, opt.grad_clip)
        utils.clip_gradient(optimizer, model,
                    opt.grad_clip_style, opt.grad_clip)
        optimizer.step()
        train_loss = loss.item()
        losses.update(loss.item())
        torch.cuda.synchronize()
        # end = time.time()
        batch_time.update(time.time() - start)

        # if iteration % opt.print_every == 0:
        #     if not sc_flag:
        #         print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
        #             .format(iteration, epoch, train_loss, end - start))
        #     else:
        #         print("iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}" \
        #             .format(iteration, epoch, np.mean(reward[:,0]), end - start))

        # logger
        if iteration % (opt.print_every*5) == 0:
            if opt.noamopt:
                opt.current_lr = optimizer.rate()
            elif opt.reduce_on_plateau:
                opt.current_lr = optimizer.current_lr
            # info_str = ' (DataTime/BatchTime: {:.3}/{:.3}) losses = {:.5}'.format(data_time.avg, batch_time.avg, losses.avg)
            # self.logger.info('Iteration ' + str(iteration) + info_str +', lr = ' +  str(self.optim.get_lr()))
            info_str = "iter {} (epoch {}), train_loss = {:.3f}, lr = {}, (DataTime/BatchTime: {:.3}/{:.3})" \
                    .format(iteration, epoch, losses.avg, opt.current_lr, data_time.avg, batch_time.avg)
            logger.info(info_str)
            
            data_time.reset()
            batch_time.reset()
            losses.reset()

        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            epoch_done = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            add_summary_value(tb_summary_writer, 'train_loss', train_loss, iteration)
            if opt.noamopt:
                opt.current_lr = optimizer.rate()
            elif opt.reduce_on_plateau:
                opt.current_lr = optimizer.current_lr
            add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)
            add_summary_value(tb_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
            if sc_flag:
                add_summary_value(tb_summary_writer, 'avg_reward', np.mean(reward[:,0]), iteration)

            # loss_history[iteration] = train_loss if not sc_flag else np.mean(reward[:,0])
            # lr_history[iteration] = opt.current_lr
            # ss_prob_history[iteration] = model.ss_prob

        # make evaluation on validation set, and save model
        if epoch_done:
            if epoch <= opt.epoch_test_more and epoch % opt.epoch_eval_every == 0:
                val_loss, predictions, lang_stats, \
                optimizer, infos, tb_summary_writer, best_val_score = evaluate(dp_model, model, epoch, loader, iteration, \
                                                                            optimizer, infos, tb_summary_writer, best_val_score)
            elif epoch > opt.epoch_test_more and epoch % 2 == 0:
                val_loss, predictions, lang_stats, \
                optimizer, infos, tb_summary_writer, best_val_score = evaluate(dp_model, model, epoch, loader, iteration, \
                                                                            optimizer, infos, tb_summary_writer, best_val_score)
        # for debug
        if iteration % opt.iteration_eval_every == 0:
            val_loss, predictions, lang_stats, \
                optimizer, infos, tb_summary_writer, best_val_score = evaluate(dp_model, model, epoch, loader, iteration, \
                                                                            optimizer, infos, tb_summary_writer, best_val_score)

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break

opt = opts.parse_opt()
logger = setup_logging(opt)
train(opt, logger)
