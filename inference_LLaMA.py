import argparse
import json, os
import torch
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM

def generate_prompt(user, ans=None):
    # if no ans, this is not history, 
    # then just add user_input to the prompt format,
    # let the bot reply the response
    prompt = f"### Instruction:\n\n{user}\n\n### Response:\n\n"

    # if there is a reply,(has history)
    # add ans to the prompt
    # prompt = f"### Instruction:\n\n{user}\n\n### Response:\n\n{ans}\n\n"
    if ans:
        prompt += f"{ans}\n\n"
    return prompt


def multi_round(args, model, tokenizer):
    # store the previous input and answer
    users = []
    bot = []
    keep_ratio = args.keep_ratio

    with torch.no_grad():
        # start the conversation
        print("欢迎使用LLaMA多轮对话, 输入信息即可获得回答, 输入restart清除聊天记录, 输入exit则退出对话。\n")
        print("LLaMA multiply round conversation instruction: type info to get answer, type 'restart' to clear conversation history, type 'exit' to end the conversation.\n")
        while True:
            # get input
            user_input = input("User:")
            # if input is restart
            if user_input == 'restart':
                # clear history
                users = []
                bot = []
                print("\n已清除聊天历史, 新的对话开始:\n")
                print("History clear, start your new conversation here:\n")
                continue

            # if user want to stop
            if user_input == 'exit':
                print("\n对话结束。\n")
                print("The conversation ends.\n")
                break
            
            # the input for this round
            input_str = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            )
            # put all history in the front of the input
            for user, ans in zip(users, bot):
                input_str += generate_prompt(user, ans)
            # put the new input in
            user_input = user_input.rstrip(' ').rstrip('\n')
            input_str += generate_prompt(user_input)

            # if the whole length is larger than how much we can keep
            if len(input_str) >= int(keep_ratio * args.max_token_num):
                # only keep last keep_ratio * args.seq_length length str
                input_str = input_str[len(input_str) - int(keep_ratio * args.max_token_num):]

            # encode the input
            input_ids = tokenizer(input_str, return_tensors="pt").input_ids.to(device)
            
            # generate output
            generate_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_token_num,
                do_sample=True,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                eos_token_id=args.eos_token_id, 
                bos_token_id=args.bos_token_id, 
                pad_token_id=args.pad_token_id
            )
            # format the output
            # decode
            whole_output = tokenizer.batch_decode(generate_ids)[0]
            # locate the rough place, [:-4] remove stop sign
            answer = whole_output[len(input_str):][:-4]

            # remove extra space, 
            answer = answer.split("\n\n")[1].rstrip(' ').rstrip('\n')
            # it is more complicated to add cut_off, so we just keep the response we need


            print(f"\nResponse: {answer}\n")
            # update users, bot
            users.append(user_input)
            bot.append(answer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # load all params
    parser.add_argument('--model', default=None, type=str, required=True)
    parser.add_argument('--tokenizer_path',default=None,type=str)
    parser.add_argument('--gpus', default="0", type=str)
    parser.add_argument('--only_cpu',action='store_true',help='only use CPU for inference')

    parser.add_argument('--max_token_num',default=2048, type=int)
    parser.add_argument('--top_p',default=0.85, type=float)
    parser.add_argument('--temperature',default=1.0, type=float)
    parser.add_argument('--repetition_penalty',default=1.2, type=float)
    parser.add_argument('--eos_token_id',default=2, type=int)
    parser.add_argument('--bos_token_id',default=1, type=int)
    parser.add_argument('--pad_token_id',default=0, type=int)
    parser.add_argument('--keep_ratio',default=0.8, type=float)

    args = parser.parse_args()

    # if only use cpu
    if args.only_cpu is True:
        args.gpus = ""
    # gpu is invisiable
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # if cuda is avaliable
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')

    # set tokenizer_path as model path, if not given
    if args.tokenizer_path is None:
        args.tokenizer_path = args.model

    # init tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=False)
    model = LlamaForCausalLM.from_pretrained(args.model, 
                                             torch_dtype=torch.float16, 
                                             device_map="auto")

    # get model vocab and tokenizer vocab
    model_vocab_size = model.get_input_embeddings().weight.size(0)
    tokenzier_vocab_size = len(tokenizer)
    if model_vocab_size!=tokenzier_vocab_size:
        print("注意: 模型词汇表和分词器词汇表不同。\nBe Careful: model vocabulary is different from tokenizer vocabulary.")

    # for cpu, fp32 may better than fp16
    if device==torch.device('cpu'):
        model.float()
    
    # make it untrained
    model.eval()

    multi_round(args, model, tokenizer)
