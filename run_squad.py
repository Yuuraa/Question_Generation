import argparse
import glob
import logging
import os
import random
import timeit

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from load_model import get_kobert_model, get_tokenizer

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForQuestionAnswering,
    DistilBertTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)
from mymodel import BertQuestionGenerator

from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult, SquadProcessor

"""HyperParameters"""
batch_size = 28
learning_rate = 5e-5
dropout_keep_prob_rate = 0.1
num_epochs = 5
overwrite_output_dir = False
local_rank = -1
train_batch_size = batch_size
output_dir = '/home/jovyan/output'
fp16 = False

do_train = True
do_eval = True
max_seq_length = 384
do_lower_case = False
tokenizer = get_tokenizer()

# 원래 args.seed 였음
def set_seed(args):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()

"""
transformers github 로부터 코드
"""
def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    # n_gpu 와 1 사이 max 해서 곱하는 거였는데, 어차피 한개니까 삭제함
    train_batch_size = per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

    # step의 최대 개수 정해져 있다면 그대로 사용하고, 아니라면 t_total 값 새로 지정
    if max_steps > 0:
        t_total = max_steps
        num_train_epochs = max_steps // (len(train_dataloader) // gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    # 옵티마이저와 스케줄 준비
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    
    #optimizer와 scheduler 정하기
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    # optimizer나 scheduler state가 이미 존재하는지 확인
    if os.path.isfile(os.path.join(model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        # 만약에 이미 저장된 상태 있다면 로드 한다
        optimizer.load_state_dict(torch.load(os.path.join(model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(model_name_or_path, "scheduler.pt")))

    # fp 16 일 경우 삭제
    # multi-gpu training (should be after apex fp16 initialization)
    # Distributed training (//) 둘 다 삭제


    # Train!
    # 학습 시작
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        train_batch_size
        * gradient_accumulation_steps,
    )
    logger.info("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    
    # Check if continuing training from a checkpoint
    # 만약 체크포인트 로부터 학습 계속할 것이라면 확인한다
    if os.path.exists(model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            # 로드 할 거라면 global_step 을 마지막 체크포인트의 global_step으로 업데이트 한다
            checkpoint_suffix = model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            # 만약에 없으면? 하는건가
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(num_train_epochs), desc="Epoch", disable=local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    # 재샌상성을 위해 추가된 코드. 
    set_seed(args)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            # 만약 이미 학습된 스텝이 있다면 건너 뛴다
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            if model_type in ["xlm", "roberta", "distilbert"]:
                del inputs["token_type_ids"]

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            # n-gpu>1 일 경우 삭제
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            # fp 16일 경우 삭제
            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if local_rank in [-1, 0] and logging_steps > 0 and global_step % logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if local_rank == -1 and evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / logging_steps, global_step)
                    logging_loss = tr_loss

                # Save model checkpoint
                # 모델의 체크포인트를 저장한다
                if local_rank in [-1, 0] and save_steps > 0 and global_step % save_steps == 0:
                    output_dir = os.path.join(output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if max_steps > 0 and global_step > max_steps:
                epoch_iterator.close()
                break
        if max_steps > 0 and global_step > max_steps:
            train_iterator.close()
            break

    if local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

"""
데이터를 가지고 평가하는 부분!
output 경로가 존재하지 않으면 안함
"""
def evaluate(args, model, tokenizer, prefix=""):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    # 만약 output_dir 존재하지 않는다면 만든다
    if not os.path.exists(output_dir) and local_rank in [-1, 0]:
        os.makedirs(output_dir)

    # eval_batch_size 어차피 ngpu = 1이므로 개당 batch size와 같음
    eval_batch_size = per_gpu_eval_batch_size

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=eval_batch_size)

    # multi-gpu evaluate
    if n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            example_indices = batch[3]
            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(output_dir, "nbest_predictions_{}.json".format(prefix))

    if version_2_with_negative:
        output_null_log_odds_file = os.path.join(output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            n_best_size,
            max_answer_length,
            do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            verbose_logging,
            version_2_with_negative,
            null_score_diff_threshold,
            tokenizer,
        )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    return results

def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    if local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    # input_dir 을 './korquad_files' 로 변경하면 될 듯
    input_dir = data_dir if data_dir else "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, model_name_or_path.split("/"))).pop(),
            str(max_seq_length),
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        if not data_dir and ((evaluate and not predict_file) or (not evaluate and not train_file)):
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

            if version_2_with_negative:
                logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

            tfds_examples = tfds.load("squad")
            examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
        else:
            processor = SquadV2Processor() if version_2_with_negative else SquadV1Processor()
            if evaluate:
                examples = processor.get_dev_examples(data_dir, filename=predict_file)
            else:
                examples = processor.get_train_examples(data_dir, filename=train_file)

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=threads,
        )

        if local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset

def main():
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    
    ## output을 저장할 곳에 이미 파일이 존재한다면 Error 출력
    if (
        os.path.exists(output_dir)
        and os.listdir(output_dir)
        and do_train
        and not overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                output_dir
            )
        )

    # Setup distant debugging if needed - 삭제함
    # Setup CUDA, GPU & distributed training - 뭐하는 건지 모름 원래는 args.local_rank 나 args.no_cuda 값에 따라 동작하였으나
    # 다른 코드에서도 여러 차례 보이던 예제로 대체함
    # args.device = device 이므로 args.device 전부 device로 대체
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup logging - 삭제

    # Set seed
    set_seed(args)

    # 프리트레인이 완료된 모델과 토크나이저를 로드한다. 나의 경우엔 위에 이미 구현 되어 있는 로드 방식을 사용함
    # args에서 굳이 가져오지 않아도 된다고 생각했음
    #Load pretrained model and tokenizer
    # local_rank 는 하나의 gpu 만을 사용할 때에는 -1로 세팅하면 되고, 아닌 경우 0 이상이다. 0은 아마도 마지막일 때 되는 듯?
    if local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    config_class, model_class, tokenizer_class = (BertConfig,BertModel, KoBertTokenizer)
    config = config_class.from_pretrained(
        config_name if config_name else model_name_or_path,
        # 기존에 args에서 받은 cache_dir 있다면 사용
        cache_dir=cache_dir if cache_dir else None,
    )
    # 위에 define 해 두었다. Kobert의 토크나이저 가져옴
    tokenizer = get_tokenizer()
    # 이 부분 수정 필요 !! from_tf 로 기존에 있던 checkpoint 가져와서 사용하려면 기준이 있어야 할 것 같다
    model = model_class.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        cache_dir=cache_dir if cache_dir else None,
    )

    # distributed 인 경우
    if local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(device)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    # fp 16 일 경우 apex import 삭제

    # Training
    if do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    # 트레이닝 하는 것이고 distributed 아닌 경우
    if do_train and (local_rank == -1):
        # Create output directory if needed
        if not os.path.exists(output_dir) and local_rank in [-1, 0]:
            os.makedirs(output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(output_dir, "training_args.bin"))

        # 내가 파인튜닝 한 것으로부터 모델 가져옴
        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(output_dir)  # , force_download=True)
        # 내가 한 토크나이저가 output_dir에 있다면 가지고 옴
        tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=args.do_lower_case)
        model.to(device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if do_eval and local_rank in [-1, 0]:
        if do_train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [output_dir]
            if eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(glob.glob(output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )
                logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
        else:
            logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
            checkpoints = [model_name_or_path]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)  # , force_download=True)
            model.to(device)

            # Evaluate
            result = evaluate(args, model, tokenizer, prefix=global_step)

            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

    logger.info("Results: {}".format(results))

    return results

# import 해서 쓰는 것이 아닌 경우 메인 함수를 실행함
if __name__ == "__main__":
    main()