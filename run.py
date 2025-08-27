from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()

prompts = [
    "", # 빈 문자열
    "\u200b" * 500, # zero-width space (spacing 없는 공백)
    ".", # 마침표
    "\n", # 줄바꿈
    "?", # 물음표
]

num_generations = 3  # 프롬프트별 생성 횟수

print("\n============================== 실험 시작 ==============================")
print("각 프롬프트별로 여러 번 문장을 생성하여 입력 민감성을 관찰합니다.\n")
print(f"{'No.':<4} {'입력 프롬프트':<30} {'입력토큰':<6} {'생성#':<6} 결과")
print("=" * 100)
for i, prompt in enumerate(prompts, 1):
    if prompt == "":
        input_ids = torch.tensor([[tokenizer.bos_token_id]])
    else:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_token_count = input_ids.shape[1]
    prompt_disp = repr(prompt)
    print(f"\n[{i}] 입력 프롬프트: {prompt_disp}")
    print(f"    (토큰 수: {input_token_count})")
    print("    " + "-" * 90)
    for n in range(1, num_generations + 1):
        attention_mask = torch.ones_like(input_ids)
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=30,
            do_sample=True,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"    [생성 {n}] {generated_text}")
    print("    " + "=" * 90)
print("\n==============================  실험 끝  ==============================")
