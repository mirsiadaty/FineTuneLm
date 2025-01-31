# FineTuneLm

Fine-tune LLM using: huggingface TRL SFTTrainer, PEFT, BNB

LLM: "mistralai/Mixtral-8x7B-Instruct-v0.1"

Task: generate executable SQL for the given 'plain English' request

The following is a snippet of the jupyer code file.

Here are a ferw examples from the trainset, to be used to fine-tune a generic LM for SQL generation:

```
{'messages': [[{'content': 'You are an text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA.\nSCHEMA:\nCREATE TABLE table_name_90 (team_1 VARCHAR)',
    'role': 'system'},
   {'content': 'Name the 2nd leg for team 1 of hamburg', 'role': 'user'},
   {'content': 'SELECT 2 AS nd_leg FROM table_name_90 WHERE team_1 = "hamburg"',
    'role': 'assistant'}],
  [{'content': 'You are an text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA.\nSCHEMA:\nCREATE TABLE table_name_54 (season VARCHAR, lead VARCHAR, third VARCHAR)',
    'role': 'system'},
   {'content': 'what is the season when the lead is john shuster and third is shawn rojeski?',
    'role': 'user'},
   {'content': 'SELECT season FROM table_name_54 WHERE lead = "john shuster" AND third = "shawn rojeski"',
    'role': 'assistant'}],
  [{'content': 'You are an text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA.\nSCHEMA:\nCREATE TABLE table_name_75 (category VARCHAR, director VARCHAR)',
    'role': 'system'},
   {'content': 'Tell me the category of na director', 'role': 'user'},
   {'content': 'SELECT category FROM table_name_75 WHERE director = "na"',
    'role': 'assistant'}],
  [{'content': 'You are an text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA.\nSCHEMA:\nCREATE TABLE table_name_25 (directed___undirected VARCHAR, induced___non_induced VARCHAR, name VARCHAR)',
    'role': 'system'},
   {'content': 'What is the directed/undirected of fpf (mavisto), which has an induced/non-induced of induced?',
    'role': 'user'},
   {'content': 'SELECT directed___undirected FROM table_name_25 WHERE induced___non_induced = "induced" AND name = "fpf (mavisto)"',
    'role': 'assistant'}],
  [{'content': 'You are an text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA.\nSCHEMA:\nCREATE TABLE table_name_10 (poll VARCHAR, wk_13 VARCHAR, wk_10 VARCHAR, wk_2 VARCHAR)',
    'role': 'system'},
   {'content': 'Which poll had a week 10 larger than 2, a week 2 of exactly 12, and a week 13 of 8?',
    'role': 'user'},
   {'content': 'SELECT poll FROM table_name_10 WHERE wk_10 > 2 AND wk_2 = "12" AND wk_13 = 8',
    'role': 'assistant'}]]}
```

We start with a MOE mixture-of-experts model from Mistral.ai and tune it for a short period of time, improving the loss:

```
model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

[67/67 2:56:49, Epoch 0/1]
Step	Training Loss
10	0.872200
20	0.526700
30	0.493400
40	0.465800
50	0.456400
60	0.448800

```

The tuned neural layers (and not the entire model), i.e. the LoRA Adapters, are saved to disk:

```
total 16165678
-rwxrwxrwx 1 ghtw30s ghtw30s        4920 Feb 11 18:26 training_args.bin
-rwxrwxrwx 1 ghtw30s ghtw30s     1795677 Feb 11 18:26 tokenizer.json
-rwxrwxrwx 1 ghtw30s ghtw30s      493443 Feb 11 18:26 tokenizer.model
-rwxrwxrwx 1 ghtw30s ghtw30s          51 Feb 11 18:26 added_tokens.json
-rwxrwxrwx 1 ghtw30s ghtw30s         557 Feb 11 18:26 special_tokens_map.json
-rwxrwxrwx 1 ghtw30s ghtw30s        1606 Feb 11 18:26 tokenizer_config.json
-rwxrwxrwx 1 ghtw30s ghtw30s         684 Feb 11 18:26 adapter_config.json
-rwxrwxrwx 1 ghtw30s ghtw30s 16551333320 Feb 11 18:26 adapter_model.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s        5112 Feb 11 18:24 README.md
drwxrwxrwx 1 ghtw30s ghtw30s         408 Feb 11 17:15 checkpoint-67
drwxrwxrwx 1 ghtw30s ghtw30s         216 Feb 11 14:16 runs
```

Alternatively, we saved the entire model where the newly tuned QLoRA Adapter layers are also inserted.

```
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM

# Load PEFT model on CPU
config = PeftConfig.from_pretrained(args.output_dir)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(args.output_dir)

#
model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(model, args.output_dir)
model = AutoPeftModelForCausalLM.from_pretrained(
    args.output_dir,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

# Merge LoRA and base model and save
merged_model = model.merge_and_unload()
merged_model.save_pretrained(args.output_dir,safe_serialization=True, max_shard_size="2GB")

87G	/media/ghtw30s/SSD-PUT/mss20230718/LLM/EXECUTE_ExecuteOnAsusG4/FineTune/models/MergedLora_20240211193450_1707698090_754
Sun Feb 11 07:48:09 PM EST 2024
total 91216713
-rwxrwxrwx 1 ghtw30s ghtw30s      92658 Feb 11 19:48 model.safetensors.index.json
-rwxrwxrwx 1 ghtw30s ghtw30s  614507328 Feb 11 19:48 model-00048-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1996490952 Feb 11 19:47 model-00047-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1963019128 Feb 11 19:47 model-00046-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1963019120 Feb 11 19:47 model-00045-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1996490952 Feb 11 19:47 model-00044-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1963019128 Feb 11 19:46 model-00043-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1963019120 Feb 11 19:46 model-00042-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1996490952 Feb 11 19:46 model-00041-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1963019128 Feb 11 19:46 model-00040-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1963019120 Feb 11 19:46 model-00039-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1996490952 Feb 11 19:45 model-00038-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1963019128 Feb 11 19:45 model-00037-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1963019120 Feb 11 19:44 model-00036-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1996490952 Feb 11 19:44 model-00035-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1963019128 Feb 11 19:44 model-00034-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1963019120 Feb 11 19:44 model-00033-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1963002512 Feb 11 19:43 model-00032-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1996507568 Feb 11 19:43 model-00031-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1963019128 Feb 11 19:42 model-00030-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1963019120 Feb 11 19:42 model-00029-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1996490952 Feb 11 19:42 model-00028-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1963019128 Feb 11 19:42 model-00027-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1963019120 Feb 11 19:42 model-00026-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1996490952 Feb 11 19:41 model-00025-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1963019128 Feb 11 19:41 model-00024-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1963019120 Feb 11 19:41 model-00023-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1996490952 Feb 11 19:40 model-00022-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1963019128 Feb 11 19:40 model-00021-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1963019120 Feb 11 19:40 model-00020-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1996490952 Feb 11 19:40 model-00019-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1963019128 Feb 11 19:40 model-00018-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1963019120 Feb 11 19:39 model-00017-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1996490952 Feb 11 19:39 model-00016-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1963019104 Feb 11 19:39 model-00015-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1963019096 Feb 11 19:39 model-00014-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1996490936 Feb 11 19:38 model-00013-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1963019104 Feb 11 19:38 model-00012-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1963019096 Feb 11 19:38 model-00011-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1996490936 Feb 11 19:38 model-00010-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1963019104 Feb 11 19:37 model-00009-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1963019096 Feb 11 19:37 model-00008-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1963002488 Feb 11 19:37 model-00007-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1996507544 Feb 11 19:37 model-00006-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1963019104 Feb 11 19:37 model-00005-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1963019096 Feb 11 19:36 model-00004-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1996490936 Feb 11 19:36 model-00003-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1963019104 Feb 11 19:36 model-00002-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s 1990281696 Feb 11 19:36 model-00001-of-00048.safetensors
-rwxrwxrwx 1 ghtw30s ghtw30s        111 Feb 11 19:36 generation_config.json
-rwxrwxrwx 1 ghtw30s ghtw30s        773 Feb 11 19:36 config.json
```



