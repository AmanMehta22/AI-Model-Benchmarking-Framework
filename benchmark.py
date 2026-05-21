import json
import time
import gc
import os
import random
import warnings
import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================
# PROFESSIONAL MULTI-MODEL AGENT BENCHMARK FRAMEWORK
# =========================================================

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    # Enable TF32 for better performance on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# -----------------------------
# Device Configuration
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_DTYPE = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else torch.float16 if DEVICE == "cuda" else torch.float32

print("=" * 90)
print("🚀 PROFESSIONAL AGENT BENCHMARK FRAMEWORK")
print("=" * 90)
print(f"Device: {DEVICE}")
print(f"Compute dtype: {COMPUTE_DTYPE}")

if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")  # FIXED: total_mem → total_memory
    print(f"TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")

# =========================================================
# MODEL REGISTRY
# =========================================================

MODELS = {
    "LFM-2.5-1.2B": "LiquidAI/LFM2.5-1.2B-Instruct",
    "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.2",
    "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen2.5-3B": "Qwen/Qwen2.5-3B-Instruct",
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct"
}

# =========================================================
# BALANCED TEST SUITE
# =========================================================

TEST_PROMPTS = [
    # --- WEATHER TOOL SELECTION (Prompts 1-12) ---
    {"id": 1, "category": "tool_use", "prompt": "Check the weather forecast for New Delhi.", "expected_tool": "get_weather"},
    {"id": 2, "category": "tool_use", "prompt": "I am traveling to Ranchi tomorrow morning. Can you pull up the live atmospheric conditions and forecast for that region?", "expected_tool": "get_weather"},
    {"id": 3, "category": "tool_use", "prompt": "Check if there are any precipitation alerts or severe weather updates active for Mumbai today.", "expected_tool": "get_weather"},
    {"id": 4, "category": "tool_use", "prompt": "Look up the humidity levels and wind speed metrics for New Delhi right now.", "expected_tool": "get_weather"},
    {"id": 5, "category": "tool_use", "prompt": "What is the expected rainfall pattern for Punjab over the next 48 hours?", "expected_tool": "get_weather"},
    {"id": 6, "category": "tool_use", "prompt": "Is there a ground frost warning active for Haryana fields tonight?", "expected_tool": "get_weather"},
    {"id": 7, "category": "tool_use", "prompt": "Pull up the live ambient temperature and moisture index for Patna right now.", "expected_tool": "get_weather"},
    {"id": 8, "category": "tool_use", "prompt": "Check the current wind direction and storm speed parameters for coastal Odisha.", "expected_tool": "get_weather"},
    {"id": 9, "category": "tool_use", "prompt": "What are the atmospheric climate readings for Nagpur agricultural zones currently?", "expected_tool": "get_weather"},
    {"id": 10, "category": "tool_use", "prompt": "Are there any active heatwave advisories issued for Rajasthan districts today?", "expected_tool": "get_weather"},
    {"id": 11, "category": "tool_use", "prompt": "Fetch the humidity index and dew point forecast for Hyderabad over this weekend.", "expected_tool": "get_weather"},
    {"id": 12, "category": "tool_use", "prompt": "Please extract the live radar rainfall metrics for the Chennai farming perimeter.", "expected_tool": "get_weather"},

    # --- SEARCH_FILES TOOL SELECTION (Prompts 13-24) ---
    {"id": 13, "category": "tool_use", "prompt": "Search my local storage for files related to mustard crop reports.", "expected_tool": "search_files"},
    {"id": 14, "category": "tool_use", "prompt": "Find PDF documents mentioning wheat disease treatment.", "expected_tool": "search_files"},
    {"id": 15, "category": "tool_use", "prompt": "Could you browse my directory and pull up any spreadsheet discussing organic pest control methods?", "expected_tool": "search_files"},
    {"id": 16, "category": "tool_use", "prompt": "Scan through the local drive server to see if we have saved any files regarding tractor maintenance logs.", "expected_tool": "search_files"},
    {"id": 17, "category": "tool_use", "prompt": "Please find the PDF manual that contains soil testing guidelines on my storage system.", "expected_tool": "search_files"},
    {"id": 18, "category": "tool_use", "prompt": "Search inside the folder system for advisory files covering aphid infestation rogor spray dosage.", "expected_tool": "search_files"},
    {"id": 19, "category": "tool_use", "prompt": "Locate any word document matching coconut flower drop treatment schedule parameters.", "expected_tool": "search_files"},
    {"id": 20, "category": "tool_use", "prompt": "Check my local files for early ahu rice variety sourcing logs from ATIC Jorhat.", "expected_tool": "search_files"},
    {"id": 21, "category": "tool_use", "prompt": "Find the excel sheet listing standard fertilizer ratios for sali crop configurations.", "expected_tool": "search_files"},
    {"id": 22, "category": "tool_use", "prompt": "Pull up the text file detailing biomanure preparation training contacts in my registry.", "expected_tool": "search_files"},
    {"id": 23, "category": "tool_use", "prompt": "Search local storage files for cauliflower damping off saaf treatment protocols.", "expected_tool": "search_files"},
    {"id": 24, "category": "tool_use", "prompt": "Locate a file named kisan_credit_card_sbi_recovery_guidelines inside my folders.", "expected_tool": "search_files"},

    # --- RESTRAINT / TRAP SELECTION (Prompts 25-36) ---
    {"id": 25, "category": "restraint", "prompt": "The weather looks pleasant today. Should I go for a walk?", "expected_tool": "none"},
    {"id": 26, "category": "restraint", "prompt": "Write a short note on agricultural automation.", "expected_tool": "none"},
    {"id": 27, "category": "restraint", "prompt": "Explain the importance of irrigation systems.", "expected_tool": "none"},
    {"id": 28, "category": "restraint", "prompt": "Farming practices have shifted over the last century. What are the key differences between traditional and modern agriculture?", "expected_tool": "none"},
    {"id": 29, "category": "restraint", "prompt": "It has been raining continuously for three days here. Do you think this will cause waterlogging in my field?", "expected_tool": "none"},
    {"id": 30, "category": "restraint", "prompt": "Can you summarize the biological process of photosynthesis in simple terms for a student?", "expected_tool": "none"},
    {"id": 31, "category": "restraint", "prompt": "My crops are turning slightly yellow. Should I consider adding more nitrogen-based fertilizer next week?", "expected_tool": "none"},
    {"id": 32, "category": "restraint", "prompt": "Explain how drip irrigation conserves more water compared to surface flood irrigation protocols.", "expected_tool": "none"},
    {"id": 33, "category": "restraint", "prompt": "Draft an introductory paragraph for a research paper analyzing sustainable farming patterns in Asia.", "expected_tool": "none"},
    {"id": 34, "category": "restraint", "prompt": "Historically, how did ancient civilizations predict weather variations before the invention of modern meteorological tools?", "expected_tool": "none"},
    {"id": 35, "category": "restraint", "prompt": "I need help understanding how filing systems work. Can you explain the difference between a physical filing cabinet and cloud storage?", "expected_tool": "none"},
    {"id": 36, "category": "restraint", "prompt": "Give me a step-by-step checklist on how to properly organize agricultural files inside a computer system manually.", "expected_tool": "none"}
]

# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """You are a strict routing agent with access to tools.

Available tools:
1. get_weather - Get current weather information for a location
2. search_files - Search local storage for files matching a query

Rules:
- If the user asks for live weather, forecasts, alerts, or atmospheric conditions, use get_weather.
- If the user asks to scan, search, browse, locate, or find local files or folders, use search_files.
- If neither tool is needed, respond with none.
- Return exactly one JSON object and nothing else.

Output format:
{"tool": "get_weather"}
{"tool": "search_files"}
{"tool": "none"}

Do not explain or add extra text."""

# =========================================================
# RESPONSE PARSER
# =========================================================

def parse_tool_response(response_text):
    """
    Robust structured tool parser.
    Extracts JSON from model output and validates tool name.
    """
    if not response_text:
        return "invalid"

    try:
        # Find first and last curly braces
        match_start = response_text.find("{")
        match_end = response_text.rfind("}") + 1

        if match_start == -1 or match_end <= match_start:
            return "invalid"

        json_block = response_text[match_start:match_end]

        # Clean common issues
        json_block = json_block.strip()

        # Parse JSON
        parsed = json.loads(json_block)

        if not isinstance(parsed, dict):
            return "invalid"

        tool = parsed.get("tool")

        if tool is None:
            return "invalid"

        # Normalize to lowercase string
        tool = str(tool).lower().strip()

        # Validate against known tools
        valid_tools = ["get_weather", "search_files", "none"]

        if tool in valid_tools:
            return tool
        else:
            return "invalid"

    except (json.JSONDecodeError, ValueError, AttributeError):
        return "invalid"

# =========================================================
# TOKENIZER SETUP HELPER
# =========================================================

def setup_tokenizer(tokenizer):
    """Configure tokenizer with proper padding and special tokens."""
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Ensure padding side is left for generation
    tokenizer.padding_side = "left"
    
    return tokenizer

# =========================================================
# CHAT TEMPLATE HELPER
# =========================================================

def format_prompt(tokenizer, messages):
    """Format messages using chat template with fallback."""
    try:
        # Try modern chat template
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return formatted
    except (TypeError, AttributeError):
        pass
    
    try:
        # Fallback 1: without add_generation_prompt
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False
        )
        return formatted
    except (TypeError, AttributeError):
        pass
    
    # Fallback 2: Manual formatting
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            formatted += f"System: {content}\n\n"
        elif role == "user":
            formatted += f"User: {content}\n\n"
        elif role == "assistant":
            formatted += f"Assistant: {content}\n\n"
    formatted += "Assistant: "
    
    return formatted

# =========================================================
# WARMUP INFERENCE
# =========================================================

def perform_warmup(model, tokenizer, device):
    """Perform warmup inference to initialize CUDA kernels."""
    warmup_message = [
        {"role": "user", "content": "Hello"}
    ]
    
    formatted = format_prompt(tokenizer, warmup_message)
    
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    
    # Synchronize after warmup
    if DEVICE == "cuda":
        torch.cuda.synchronize()

# =========================================================
# BENCHMARK EXECUTION
# =========================================================

benchmark_results = []

for model_name, hf_model_id in MODELS.items():

    print("\n" + "=" * 90)
    print(f"⏳ Loading Model: {model_name}")
    print(f"📦 Hugging Face ID: {hf_model_id}")
    print("=" * 90)

    try:
        # -------------------------------------------------
        # Load Tokenizer
        # -------------------------------------------------
        print("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            hf_model_id,
            trust_remote_code=True,
            use_fast=False  # FIXED: use_fast=False for compatibility (especially LiquidAI)
        )

        tokenizer = setup_tokenizer(tokenizer)
        print(f"  ✅ Tokenizer loaded (vocab size: {len(tokenizer)})")

        # -------------------------------------------------
        # Load Model
        # -------------------------------------------------
        print("  Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            torch_dtype=COMPUTE_DTYPE,
            device_map="auto" if DEVICE == "cuda" else None,
            trust_remote_code=True,  # FIXED: CRITICAL - Added trust_remote_code=True
            low_cpu_mem_usage=True
        )

        if DEVICE == "cpu":
            model = model.to(DEVICE)

        model.eval()
        
        # FIXED: Removed resize_token_embeddings call - using pad_token assignment instead
        # This prevents AttributeError with LiquidAI models
        
        print(f"  ✅ Model loaded")

        # -------------------------------------------------
        # Warmup (FIXED: Added warmup before timing)
        # -------------------------------------------------
        print("  Performing warmup inference...")
        perform_warmup(model, tokenizer, model.device)
        print("  ✅ Warmup complete")

        # -------------------------------------------------
        # Metrics Initialization
        # -------------------------------------------------
        tool_correct = 0
        restraint_correct = 0
        wrong_tool_calls = 0
        invalid_responses = 0

        total_latency_ms = 0
        total_tokens_generated = 0
        total_prompt_tokens = 0
        
        # Track truncation warnings
        truncation_warnings = 0

        # -------------------------------------------------
        # Run Evaluation
        # -------------------------------------------------
        for sample in TEST_PROMPTS:
            print(f"\n  ▶ Test {sample['id']}/{len(TEST_PROMPTS)}: [{sample['category']}] {sample['prompt'][:60]}...")

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": sample["prompt"]}
            ]

            # Format prompt
            formatted_prompt = format_prompt(tokenizer, messages)

            # Tokenize
            inputs = tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )

            # Check for truncation
            if inputs["input_ids"].shape[1] >= 2048:
                truncation_warnings += 1
                print(f"    ⚠️ Warning: Input may be truncated (length: {inputs['input_ids'].shape[1]})")

            # FIXED: Use model.device instead of hardcoded .cuda()
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            prompt_token_count = inputs["input_ids"].shape[1]
            total_prompt_tokens += prompt_token_count

            # ---------------------------------------------
            # Inference with Proper Timing
            # ---------------------------------------------
            
            # FIXED: Synchronize before timing
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()

            with torch.no_grad():
                try:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=30,
                        do_sample=False,  # FIXED: Removed temperature parameter entirely
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True
                    )
                except Exception as gen_err:
                    print(f"    ⚠️ Generation error: {gen_err}")
                    # Fallback: minimal generation
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=20,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )

            # FIXED: Synchronize after generation
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            total_latency_ms += latency_ms

            # ---------------------------------------------
            # Decode Output
            # ---------------------------------------------
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            
            # Prevent counting pad tokens in generation
            if tokenizer.pad_token_id is not None:
                generated_tokens = generated_tokens[generated_tokens != tokenizer.pad_token_id]
            
            num_generated = len(generated_tokens)
            total_tokens_generated += num_generated

            response = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            ).strip()

            predicted_tool = parse_tool_response(response)
            expected_tool = sample["expected_tool"]

            # Debug output
            print(f"    Generated: {response[:100]}")
            print(f"    Predicted tool: {predicted_tool}")
            print(f"    Expected tool: {expected_tool or 'none'}")

            # ---------------------------------------------
            # FIXED: Scoring Logic - No double punishment
            # ---------------------------------------------
            if predicted_tool == "invalid":
                invalid_responses += 1
                # FIXED: Removed wrong_tool_calls increment here
                # Invalid responses are tracked separately and penalized via valid_response_rate
                print(f"    ❌ Invalid response")
            elif sample["category"] == "tool_use":
                if predicted_tool == expected_tool:
                    tool_correct += 1
                    print(f"    ✅ Correct tool call")
                else:
                    wrong_tool_calls += 1
                    print(f"    ❌ Wrong tool called (got: {predicted_tool})")
            elif sample["category"] == "restraint":
                if predicted_tool == "none":
                    restraint_correct += 1
                    print(f"    ✅ Correctly restrained")
                else:
                    wrong_tool_calls += 1
                    print(f"    ❌ Incorrectly called tool: {predicted_tool}")

        # =================================================
        # FINAL METRICS CALCULATION
        # =================================================
        tool_total = len([x for x in TEST_PROMPTS if x["category"] == "tool_use"])
        restraint_total = len([x for x in TEST_PROMPTS if x["category"] == "restraint"])

        action_score = tool_correct / tool_total if tool_total > 0 else 0.0
        restraint_score = restraint_correct / restraint_total if restraint_total > 0 else 0.0
        
        # FIXED: Wrong tool penalty now only counts actual wrong calls (not invalid)
        wrong_tool_penalty = max(0.0, 1.0 - (wrong_tool_calls / len(TEST_PROMPTS)))
        valid_response_rate = (len(TEST_PROMPTS) - invalid_responses) / len(TEST_PROMPTS)

        avg_latency_ms = total_latency_ms / len(TEST_PROMPTS)
        
        # FIXED: Protected division by zero
        if total_latency_ms > 0:
            throughput = total_tokens_generated / (total_latency_ms / 1000)
        else:
            throughput = 0.0

        # Weighted Composite Agent Score
        agent_score = (
            (action_score * 0.40) +
            (restraint_score * 0.30) +
            (wrong_tool_penalty * 0.20) +
            (valid_response_rate * 0.10)
        )

        # =================================================
        # LOG RESULTS
        # =================================================
        result = {
            "Model": model_name,
            "HF Model ID": hf_model_id,
            "Action Score": round(action_score, 3),
            "Restraint Score": round(restraint_score, 3),
            "Wrong Tool Calls": wrong_tool_calls,
            "Invalid Responses": invalid_responses,
            "Valid Response Rate": round(valid_response_rate, 3),
            "Agent Score": round(agent_score, 3),
            "Avg Latency (ms)": round(avg_latency_ms, 2),
            "Throughput (tok/s)": round(throughput, 2),
            "Total Tokens Generated": total_tokens_generated,
            "Avg Prompt Tokens": total_prompt_tokens // len(TEST_PROMPTS),
            "Truncation Warnings": truncation_warnings
        }

        benchmark_results.append(result)

        print(f"\n  {'='*80}")
        print(f"  📊 Model Summary: {model_name}")
        print(f"  {'='*80}")
        print(f"  ✅ Action Score:      {action_score:.3f} ({tool_correct}/{tool_total})")
        print(f"  ✅ Restraint Score:   {restraint_score:.3f} ({restraint_correct}/{restraint_total})")
        print(f"  ❌ Wrong Calls:       {wrong_tool_calls}")
        print(f"  ⚠️  Invalid:          {invalid_responses}")
        print(f"  📋 Valid Rate:        {valid_response_rate:.3f}")
        print(f"  ⚠️  Truncations:      {truncation_warnings}")
        print(f"  🎯 Agent Score:       {agent_score:.3f}")
        print(f"  ⚡ Avg Latency:       {avg_latency_ms:.2f} ms")
        print(f"  🚀 Throughput:        {throughput:.2f} tok/s")

        # =================================================
        # MEMORY CLEANUP
        # =================================================
        del model
        del tokenizer
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    except Exception as e:
        print(f"\n  ❌ CRITICAL ERROR Benchmarking {model_name}")
        print(f"  Error: {str(e)}")
        print(f"  Error type: {type(e).__name__}")

        # Add placeholder result for failed model
        benchmark_results.append({
            "Model": model_name,
            "HF Model ID": hf_model_id,
            "Action Score": 0.0,
            "Restraint Score": 0.0,
            "Wrong Tool Calls": len(TEST_PROMPTS),
            "Invalid Responses": len(TEST_PROMPTS),
            "Valid Response Rate": 0.0,
            "Agent Score": 0.0,
            "Avg Latency (ms)": 0.0,
            "Throughput (tok/s)": 0.0,
            "Total Tokens Generated": 0,
            "Avg Prompt Tokens": 0,
            "Truncation Warnings": 0
        })

        # Cleanup in case of partial load
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        continue

# =========================================================
# FINAL REPORT GENERATION
# =========================================================

print("\n" + "=" * 90)
print("🎉 BENCHMARK COMPLETE - GENERATING REPORT")
print("=" * 90)

# Create DataFrame
report_df = pd.DataFrame(benchmark_results)

# Sort by Agent Score
report_df = report_df.sort_values(
    by="Agent Score",
    ascending=False
).reset_index(drop=True)

# Add ranking
report_df.index += 1
report_df.index.name = "Rank"

# =========================================================
# DISPLAY FINAL TABLE
# =========================================================

print("\n" + "=" * 90)
print("📊 FINAL BENCHMARK RESULTS")
print("=" * 90)

# Display formatted table
display_columns = [
    "Model",
    "Agent Score",
    "Action Score",
    "Restraint Score",
    "Valid Response Rate",
    "Wrong Tool Calls",
    "Invalid Responses",
    "Avg Latency (ms)",
    "Throughput (tok/s)"
]

print(report_df[display_columns].to_string())

# =========================================================
# EXPORT RESULTS
# =========================================================

import datetime

# Create results folder if it doesn't exist
results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"📁 Created results directory: {results_dir}/")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# CSV Export
csv_file = os.path.join(results_dir, f"agent_benchmark_results_{timestamp}.csv")
report_df.to_csv(csv_file)
print(f"\n📁 Results saved to: {csv_file}")

# JSON Export (more detailed)
json_file = os.path.join(results_dir, f"agent_benchmark_results_{timestamp}.json")
report_df.to_json(json_file, orient="records", indent=2)
print(f"📁 Detailed results saved to: {json_file}")

# =========================================================
# PERFORMANCE SUMMARY
# =========================================================

print("\n" + "=" * 90)
print("📈 PERFORMANCE SUMMARY")
print("=" * 90)

if len(report_df) > 0 and report_df["Agent Score"].max() > 0:
    best_model = report_df.iloc[0]
    print(f"🏆 Best Performing Model: {best_model['Model']}")
    print(f"🎯 Agent Score: {best_model['Agent Score']:.3f}")
    print(f"⚡ Latency: {best_model['Avg Latency (ms)']:.2f} ms")
    print(f"🚀 Throughput: {best_model['Throughput (tok/s)']:.2f} tok/s")

    # Calculate averages (excluding failed models)
    working_models = report_df[report_df["Agent Score"] > 0]
    if len(working_models) > 0:
        avg_agent_score = working_models["Agent Score"].mean()
        avg_latency = working_models["Avg Latency (ms)"].mean()
        avg_throughput = working_models["Throughput (tok/s)"].mean()

        print(f"\n📊 Average Across Working Models ({len(working_models)}/{len(report_df)}):")
        print(f"   Agent Score: {avg_agent_score:.3f}")
        print(f"   Latency: {avg_latency:.2f} ms")
        print(f"   Throughput: {avg_throughput:.2f} tok/s")

print("\n" + "=" * 90)
print("✅ BENCHMARK EXECUTION COMPLETE")
print("=" * 90)