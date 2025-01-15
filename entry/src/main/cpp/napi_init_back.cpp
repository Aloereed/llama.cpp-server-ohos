#include "napi/native_api.h"
#include "llama/arm64-v8a/include/llama-cpp.h"
#include "hilog/log.h"
#include "llama/arm64-v8a/include/arg.h"
#include "llama/arm64-v8a/include/common.h"
#include "llama/arm64-v8a/include/console.h"
#include "llama/arm64-v8a/include/log.h"
#include "llama/arm64-v8a/include/sampling.h"
#include <future>
#undef LOG_DOMAIN
#undef LOG_TAG
#define LOG_DOMAIN 0x3200  // 全局domain宏，标识业务领域
#define LOG_TAG "MY_TAG"   // 全局tag宏，标识模块日志tag

#include <cassert>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static llama_context           ** g_ctx;
static llama_model             ** g_model;
static common_sampler          ** g_smpl;
static common_params            * g_params;
static std::vector<llama_token> * g_input_tokens;
static std::ostringstream       * g_output_ss;
static std::vector<llama_token> * g_output_tokens;
static bool is_interacting  = false;
static bool need_insert_eot = false;

static void print_usage(int argc, char ** argv) {
    (void) argc;

    LOG("\nexample usage:\n");
    LOG("\n  text generation:     %{public}s -m your_model.gguf -p \"I believe the meaning of life is\" -n 128\n", argv[0]);
    LOG("\n  chat (conversation): %{public}s -m your_model.gguf -p \"You are a helpful assistant\" -cnv\n", argv[0]);
    LOG("\n");
}

static bool file_exists(const std::string & path) {
    std::ifstream f(path.c_str());
    return f.good();
}

static bool file_is_empty(const std::string & path) {
    std::ifstream f;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    f.open(path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    return f.tellg() == 0;
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (!is_interacting && g_params->interactive) {
            is_interacting  = true;
            need_insert_eot = true;
        } else {
            console::cleanup();
            LOG("\n");
            common_perf_print(*g_ctx, *g_smpl);

            // make sure all logs are flushed
            LOG("Interrupted by user\n");
            common_log_pause(common_log_main());

            _exit(130);
        }
    }
}
#endif

static std::string chat_add_and_format(struct llama_model * model, std::vector<common_chat_msg> & chat_msgs, const std::string & role, const std::string & content) {
    common_chat_msg new_msg{role, content};
    auto formatted = common_chat_format_single(model, g_params->chat_template, chat_msgs, new_msg, role == "user");
    chat_msgs.push_back({role, content});
    OH_LOG_ERROR(LOG_APP,"formatted: '%{public}s'\n", formatted.c_str());
    return formatted;
}

int main_function(int argc, char ** argv, std::string& result) {
    common_params params;
    g_params = &params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MAIN, print_usage)) {
        return 1;
    }

    common_init();

    auto & sparams = params.sampling;

    // save choice to use color for later
    // (note for later: this is a slightly awkward choice)
    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });

    if (params.logits_all) {
        OH_LOG_ERROR(LOG_APP,"************\n");
        OH_LOG_ERROR(LOG_APP,"%{public}s: please use the 'perplexity' tool for perplexity calculations\n", __func__);
        OH_LOG_ERROR(LOG_APP,"************\n\n");

        return 0;
    }

    if (params.embedding) {
        OH_LOG_ERROR(LOG_APP,"************\n");
        OH_LOG_ERROR(LOG_APP,"%{public}s: please use the 'embedding' tool for embedding calculations\n", __func__);
        OH_LOG_ERROR(LOG_APP,"************\n\n");

        return 0;
    }

    if (params.n_ctx != 0 && params.n_ctx < 8) {
        OH_LOG_ERROR(LOG_APP,"%{public}s: warning: minimum context size is 8, using minimum size.\n", __func__);
        params.n_ctx = 8;
    }

    if (params.rope_freq_base != 0.0) {
        OH_LOG_ERROR(LOG_APP,"%{public}s: warning: changing RoPE frequency base to %g.\n", __func__, params.rope_freq_base);
    }

    if (params.rope_freq_scale != 0.0) {
        OH_LOG_ERROR(LOG_APP,"%{public}s: warning: scaling RoPE frequency by %g.\n", __func__, params.rope_freq_scale);
    }

    OH_LOG_ERROR(LOG_APP,"%{public}s: llama backend init\n", __func__);

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    common_sampler * smpl = nullptr;

    g_model = &model;
    g_ctx = &ctx;
    g_smpl = &smpl;

    std::vector<common_chat_msg> chat_msgs;

    // load the model and apply lora adapter, if any
    OH_LOG_ERROR(LOG_APP,"%{public}s: load the model and apply lora adapter, if any\n", __func__);
    common_init_result llama_init = common_init_from_params(params);

    model = llama_init.model.get();
    ctx = llama_init.context.get();

    if (model == NULL) {
        OH_LOG_ERROR(LOG_APP,"%{public}s: error: unable to load model\n", __func__);
        return 1;
    }

    OH_LOG_ERROR(LOG_APP,"%{public}s: llama threadpool init, n_threads = %d\n", __func__, (int) params.cpuparams.n_threads);

    auto * reg = ggml_backend_dev_backend_reg(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
    auto * ggml_threadpool_new_fn = (decltype(ggml_threadpool_new) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
    auto * ggml_threadpool_free_fn = (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");

    struct ggml_threadpool_params tpp_batch =
            ggml_threadpool_params_from_cpu_params(params.cpuparams_batch);
    struct ggml_threadpool_params tpp =
            ggml_threadpool_params_from_cpu_params(params.cpuparams);

    set_process_priority(params.cpuparams.priority);

    struct ggml_threadpool * threadpool_batch = NULL;
    if (!ggml_threadpool_params_match(&tpp, &tpp_batch)) {
        threadpool_batch = ggml_threadpool_new_fn(&tpp_batch);
        if (!threadpool_batch) {
            OH_LOG_ERROR(LOG_APP,"%{public}s: batch threadpool create failed : n_threads %d\n", __func__, tpp_batch.n_threads);
            return 1;
        }

        // Start the non-batch threadpool in the paused state
        tpp.paused = true;
    }

    struct ggml_threadpool * threadpool = ggml_threadpool_new_fn(&tpp);
    if (!threadpool) {
        OH_LOG_ERROR(LOG_APP,"%{public}s: threadpool create failed : n_threads %d\n", __func__, tpp.n_threads);
        return 1;
    }

    llama_attach_threadpool(ctx, threadpool, threadpool_batch);

    const int n_ctx_train = llama_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx);

    if (n_ctx > n_ctx_train) {
        OH_LOG_ERROR(LOG_APP,"%{public}s: model was trained on only %d context tokens (%d specified)\n", __func__, n_ctx_train, n_ctx);
    }

    // print chat template example in conversation mode
    if (params.conversation) {
        if (params.enable_chat_template) {
            OH_LOG_ERROR(LOG_APP,"%{public}s: chat template example:\n%{public}s\n", __func__, common_chat_format_example(model, params.chat_template).c_str());
        } else {
            OH_LOG_ERROR(LOG_APP,"%{public}s: in-suffix/prefix is specified, chat template will be disabled\n", __func__);
        }
    }

    // print system information
    {
        OH_LOG_ERROR(LOG_APP,"\n");
        OH_LOG_ERROR(LOG_APP,"%{public}s\n", common_params_get_system_info(params).c_str());
        OH_LOG_ERROR(LOG_APP,"\n");
    }

    std::string path_session = params.path_prompt_cache;
    std::vector<llama_token> session_tokens;

    if (!path_session.empty()) {
        OH_LOG_ERROR(LOG_APP,"%{public}s: attempting to load saved session from '%{public}s'\n", __func__, path_session.c_str());
        if (!file_exists(path_session)) {
            OH_LOG_ERROR(LOG_APP,"%{public}s: session file does not exist, will create.\n", __func__);
        } else if (file_is_empty(path_session)) {
            OH_LOG_ERROR(LOG_APP,"%{public}s: The session file is empty. A new session will be initialized.\n", __func__);
        } else {
            // The file exists and is not empty
            session_tokens.resize(n_ctx);
            size_t n_token_count_out = 0;
            if (!llama_state_load_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.capacity(), &n_token_count_out)) {
                OH_LOG_ERROR(LOG_APP,"%{public}s: failed to load session file '%{public}s'\n", __func__, path_session.c_str());
                return 1;
            }
            session_tokens.resize(n_token_count_out);
            OH_LOG_ERROR(LOG_APP,"%{public}s: loaded a session with prompt size of %d tokens\n", __func__, (int)session_tokens.size());
        }
    }

    const bool add_bos = llama_add_bos_token(model);
    if (!llama_model_has_encoder(model)) {
        GGML_ASSERT(!llama_add_eos_token(model));
    }

    OH_LOG_ERROR(LOG_APP,"n_ctx: %d, add_bos: %d\n", n_ctx, add_bos);

    std::vector<llama_token> embd_inp;

    {
        auto prompt = (params.conversation && params.enable_chat_template && !params.prompt.empty())
            ? chat_add_and_format(model, chat_msgs, "system", params.prompt) // format the system prompt in conversation mode
            : params.prompt;
        if (params.interactive_first || !params.prompt.empty() || session_tokens.empty()) {
            OH_LOG_ERROR(LOG_APP,"tokenize the prompt\n");
            embd_inp = common_tokenize(ctx, prompt, true, true);
        } else {
            OH_LOG_ERROR(LOG_APP,"use session tokens\n");
            embd_inp = session_tokens;
        }

        OH_LOG_ERROR(LOG_APP,"prompt: \"%{public}s\"\n", prompt.c_str());
        OH_LOG_ERROR(LOG_APP,"tokens: %{public}s\n", string_from(ctx, embd_inp).c_str());
    }

    // Should not run without any tokens
    if (embd_inp.empty()) {
        if (add_bos) {
            embd_inp.push_back(llama_token_bos(model));
            OH_LOG_ERROR(LOG_APP,"embd_inp was considered empty and bos was added: %{public}s\n", string_from(ctx, embd_inp).c_str());
        } else {
            OH_LOG_ERROR(LOG_APP,"input is empty\n");
            return -1;
        }
    }

    // Tokenize negative prompt
    if ((int) embd_inp.size() > n_ctx - 4) {
        OH_LOG_ERROR(LOG_APP,"%{public}s: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return 1;
    }

    // debug message about similarity of saved session, if applicable
    size_t n_matching_session_tokens = 0;
    if (!session_tokens.empty()) {
        for (llama_token id : session_tokens) {
            if (n_matching_session_tokens >= embd_inp.size() || id != embd_inp[n_matching_session_tokens]) {
                break;
            }
            n_matching_session_tokens++;
        }
        if (params.prompt.empty() && n_matching_session_tokens == embd_inp.size()) {
            OH_LOG_ERROR(LOG_APP,"%{public}s: using full prompt from session file\n", __func__);
        } else if (n_matching_session_tokens >= embd_inp.size()) {
            OH_LOG_ERROR(LOG_APP,"%{public}s: session file has exact match for prompt!\n", __func__);
        } else if (n_matching_session_tokens < (embd_inp.size() / 2)) {
            OH_LOG_ERROR(LOG_APP,"%{public}s: session file has low similarity to prompt (%{public}zu / %{public}zu tokens); will mostly be reevaluated\n",
                    __func__, n_matching_session_tokens, embd_inp.size());
        } else {
            OH_LOG_ERROR(LOG_APP,"%{public}s: session file matches %{public}zu / %{public}zu tokens of prompt\n",
                    __func__, n_matching_session_tokens, embd_inp.size());
        }

        // remove any "future" tokens that we might have inherited from the previous session
        llama_kv_cache_seq_rm(ctx, -1, n_matching_session_tokens, -1);
    }

    OH_LOG_ERROR(LOG_APP,"recalculate the cached logits (check): embd_inp.size() %{public}zu, n_matching_session_tokens %{public}zu, embd_inp.size() %{public}zu, session_tokens.size() %{public}zu\n",
         embd_inp.size(), n_matching_session_tokens, embd_inp.size(), session_tokens.size());

    // if we will use the cache for the full prompt without reaching the end of the cache, force
    // reevaluation of the last token to recalculate the cached logits
    if (!embd_inp.empty() && n_matching_session_tokens == embd_inp.size() && session_tokens.size() > embd_inp.size()) {
        OH_LOG_ERROR(LOG_APP,"recalculate the cached logits (do): session_tokens.resize( %{public}zu )\n", embd_inp.size() - 1);

        session_tokens.resize(embd_inp.size() - 1);
    }

    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size()) {
        params.n_keep = (int)embd_inp.size();
    } else {
        params.n_keep += add_bos; // always keep the BOS token
    }

    if (params.conversation) {
        params.interactive_first = true;
    }

    // enable interactive mode if interactive start is specified
    if (params.interactive_first) {
        params.interactive = true;
    }

    if (params.verbose_prompt) {
        OH_LOG_ERROR(LOG_APP,"%{public}s: prompt: '%{public}s'\n", __func__, params.prompt.c_str());
        OH_LOG_ERROR(LOG_APP,"%{public}s: number of tokens in prompt = %{public}zu\n", __func__, embd_inp.size());
        for (int i = 0; i < (int) embd_inp.size(); i++) {
            OH_LOG_ERROR(LOG_APP,"%6d -> '%{public}s'\n", embd_inp[i], common_token_to_piece(ctx, embd_inp[i]).c_str());
        }

        if (params.n_keep > add_bos) {
            OH_LOG_ERROR(LOG_APP,"%{public}s: static prompt based on n_keep: '", __func__);
            for (int i = 0; i < params.n_keep; i++) {
                LOG_CNT("%{public}s", common_token_to_piece(ctx, embd_inp[i]).c_str());
            }
            LOG_CNT("'\n");
        }
        OH_LOG_ERROR(LOG_APP,"\n");
    }

    // ctrl+C handling
    {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = sigint_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
    }

    if (params.interactive) {
        OH_LOG_ERROR(LOG_APP,"%{public}s: interactive mode on.\n", __func__);

        if (!params.antiprompt.empty()) {
            for (const auto & antiprompt : params.antiprompt) {
                OH_LOG_ERROR(LOG_APP,"Reverse prompt: '%{public}s'\n", antiprompt.c_str());
                if (params.verbose_prompt) {
                    auto tmp = common_tokenize(ctx, antiprompt, false, true);
                    for (int i = 0; i < (int) tmp.size(); i++) {
                        OH_LOG_ERROR(LOG_APP,"%{public}6d -> '%{public}s'\n", tmp[i], common_token_to_piece(ctx, tmp[i]).c_str());
                    }
                }
            }
        }

        if (params.input_prefix_bos) {
            OH_LOG_ERROR(LOG_APP,"Input prefix with BOS\n");
        }

        if (!params.input_prefix.empty()) {
            OH_LOG_ERROR(LOG_APP,"Input prefix: '%{public}s'\n", params.input_prefix.c_str());
            if (params.verbose_prompt) {
                auto tmp = common_tokenize(ctx, params.input_prefix, true, true);
                for (int i = 0; i < (int) tmp.size(); i++) {
                    OH_LOG_ERROR(LOG_APP,"%6d -> '%{public}s'\n", tmp[i], common_token_to_piece(ctx, tmp[i]).c_str());
                }
            }
        }

        if (!params.input_suffix.empty()) {
            OH_LOG_ERROR(LOG_APP,"Input suffix: '%{public}s'\n", params.input_suffix.c_str());
            if (params.verbose_prompt) {
                auto tmp = common_tokenize(ctx, params.input_suffix, false, true);
                for (int i = 0; i < (int) tmp.size(); i++) {
                    OH_LOG_ERROR(LOG_APP,"%6d -> '%{public}s'\n", tmp[i], common_token_to_piece(ctx, tmp[i]).c_str());
                }
            }
        }
    }

    smpl = common_sampler_init(model, sparams);
    if (!smpl) {
        OH_LOG_ERROR(LOG_APP,"%{public}s: failed to initialize sampling subsystem\n", __func__);
        return 1;
    }

    OH_LOG_ERROR(LOG_APP,"sampler seed: %{public}u\n",     common_sampler_get_seed(smpl));
    OH_LOG_ERROR(LOG_APP,"sampler params: \n%{public}s\n", sparams.print().c_str());
    OH_LOG_ERROR(LOG_APP,"sampler chain: %{public}s\n",    common_sampler_print(smpl).c_str());

    OH_LOG_ERROR(LOG_APP,"generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);

    // group-attention state
    // number of grouped KV tokens so far (used only if params.grp_attn_n > 1)
    int ga_i = 0;

    const int ga_n = params.grp_attn_n;
    const int ga_w = params.grp_attn_w;

    if (ga_n != 1) {
        GGML_ASSERT(ga_n > 0                    && "grp_attn_n must be positive");                     // NOLINT
        GGML_ASSERT(ga_w % ga_n == 0            && "grp_attn_w must be a multiple of grp_attn_n");     // NOLINT
      //GGML_ASSERT(n_ctx_train % ga_w == 0     && "n_ctx_train must be a multiple of grp_attn_w");    // NOLINT
      //GGML_ASSERT(n_ctx >= n_ctx_train * ga_n && "n_ctx must be at least n_ctx_train * grp_attn_n"); // NOLINT
        OH_LOG_ERROR(LOG_APP,"self-extend: n_ctx_train = %d, grp_attn_n = %d, grp_attn_w = %d\n", n_ctx_train, ga_n, ga_w);
    }
    OH_LOG_ERROR(LOG_APP,"\n");

    if (params.interactive) {
        const char * control_message;
        if (params.multiline_input) {
            control_message = " - To return control to the AI, end your input with '\\'.\n"
                              " - To return control without starting a new line, end your input with '/'.\n";
        } else {
            control_message = " - Press Return to return control to the AI.\n"
                              " - To return control without starting a new line, end your input with '/'.\n"
                              " - If you want to submit another line, end your input with '\\'.\n";
        }
        OH_LOG_ERROR(LOG_APP,"== Running in interactive mode. ==\n");
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
        OH_LOG_ERROR(LOG_APP,       " - Press Ctrl+C to interject at any time.\n");
#endif
        OH_LOG_ERROR(LOG_APP,       "%{public}s\n", control_message);

        is_interacting = params.interactive_first;
    }

    bool is_antiprompt        = false;
    bool input_echo           = true;
    bool display              = true;
    bool need_to_save_session = !path_session.empty() && n_matching_session_tokens < embd_inp.size();

    int n_past             = 0;
    int n_remain           = params.n_predict;
    int n_consumed         = 0;
    int n_session_consumed = 0;

    std::vector<int>   input_tokens;  g_input_tokens  = &input_tokens;
    std::vector<int>   output_tokens; g_output_tokens = &output_tokens;
    std::ostringstream output_ss;     g_output_ss     = &output_ss;
    std::ostringstream assistant_ss; // for storing current assistant message, used in conversation mode

    // the first thing we will do is to output the prompt, so set color accordingly
    console::set_display(console::prompt);
    display = params.display_prompt;

    std::vector<llama_token> embd;

    // tokenized antiprompts
    std::vector<std::vector<llama_token>> antiprompt_ids;

    antiprompt_ids.reserve(params.antiprompt.size());
    for (const std::string & antiprompt : params.antiprompt) {
        antiprompt_ids.emplace_back(::common_tokenize(ctx, antiprompt, false, true));
    }

    if (llama_model_has_encoder(model)) {
        int enc_input_size = embd_inp.size();
        llama_token * enc_input_buf = embd_inp.data();

        if (llama_encode(ctx, llama_batch_get_one(enc_input_buf, enc_input_size))) {
            OH_LOG_ERROR(LOG_APP,"%{public}s : failed to eval\n", __func__);
            return 1;
        }

        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
            decoder_start_token_id = llama_token_bos(model);
        }

        embd_inp.clear();
        embd_inp.push_back(decoder_start_token_id);
    }

    while ((n_remain != 0 && !is_antiprompt) || params.interactive) {
        // predict
        if (!embd.empty()) {
            // Note: (n_ctx - 4) here is to match the logic for commandline prompt handling via
            // --prompt or --file which uses the same value.
            int max_embd_size = n_ctx - 4;

            // Ensure the input doesn't exceed the context size by truncating embd if necessary.
            if ((int) embd.size() > max_embd_size) {
                const int skipped_tokens = (int) embd.size() - max_embd_size;
                embd.resize(max_embd_size);

                console::set_display(console::error);
                OH_LOG_ERROR(LOG_APP,"<<input too long: skipped %d token%{public}s>>", skipped_tokens, skipped_tokens != 1 ? "s" : "");
                console::set_display(console::reset);
            }

            if (ga_n == 1) {
                // infinite text generation via context shifting
                // if we run out of context:
                // - take the n_keep first tokens from the original prompt (via n_past)
                // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches

                if (n_past + (int) embd.size() >= n_ctx) {
                    if (!params.ctx_shift){
                        OH_LOG_ERROR(LOG_APP,"\n\n%{public}s: context full and context shift is disabled => stopping\n", __func__);
                        break;
                    }

                    if (params.n_predict == -2) {
                        OH_LOG_ERROR(LOG_APP,"\n\n%{public}s: context full and n_predict == -%d => stopping\n", __func__, params.n_predict);
                        break;
                    }

                    const int n_left    = n_past - params.n_keep;
                    const int n_discard = n_left/2;

                    OH_LOG_ERROR(LOG_APP,"context full, swapping: n_past = %d, n_left = %d, n_ctx = %d, n_keep = %d, n_discard = %d\n",
                            n_past, n_left, n_ctx, params.n_keep, n_discard);

                    llama_kv_cache_seq_rm (ctx, 0, params.n_keep            , params.n_keep + n_discard);
                    llama_kv_cache_seq_add(ctx, 0, params.n_keep + n_discard, n_past, -n_discard);

                    n_past -= n_discard;

                    OH_LOG_ERROR(LOG_APP,"after swap: n_past = %d\n", n_past);

                    OH_LOG_ERROR(LOG_APP,"embd: %{public}s\n", string_from(ctx, embd).c_str());

                    OH_LOG_ERROR(LOG_APP,"clear session path\n");
                    path_session.clear();
                }
            } else {
                // context extension via Self-Extend
                while (n_past >= ga_i + ga_w) {
                    const int ib = (ga_n*ga_i)/ga_w;
                    const int bd = (ga_w/ga_n)*(ga_n - 1);
                    const int dd = (ga_w/ga_n) - ib*bd - ga_w;

                    OH_LOG_ERROR(LOG_APP,"\n");
                    OH_LOG_ERROR(LOG_APP,"shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i, n_past, ib*bd, ga_i + ib*bd, n_past + ib*bd);
                    OH_LOG_ERROR(LOG_APP,"div:   [%6d, %6d] / %6d -> [%6d, %6d]\n", ga_i + ib*bd, ga_i + ib*bd + ga_w, ga_n, (ga_i + ib*bd)/ga_n, (ga_i + ib*bd + ga_w)/ga_n);
                    OH_LOG_ERROR(LOG_APP,"shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i + ib*bd + ga_w, n_past + ib*bd, dd, ga_i + ib*bd + ga_w + dd, n_past + ib*bd + dd);

                    llama_kv_cache_seq_add(ctx, 0, ga_i,                n_past,              ib*bd);
                    llama_kv_cache_seq_div(ctx, 0, ga_i + ib*bd,        ga_i + ib*bd + ga_w, ga_n);
                    llama_kv_cache_seq_add(ctx, 0, ga_i + ib*bd + ga_w, n_past + ib*bd,      dd);

                    n_past -= bd;

                    ga_i += ga_w/ga_n;

                    OH_LOG_ERROR(LOG_APP,"\nn_past_old = %d, n_past = %d, ga_i = %d\n\n", n_past + bd, n_past, ga_i);
                }
            }

            // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
            if (n_session_consumed < (int) session_tokens.size()) {
                size_t i = 0;
                for ( ; i < embd.size(); i++) {
                    if (embd[i] != session_tokens[n_session_consumed]) {
                        session_tokens.resize(n_session_consumed);
                        break;
                    }

                    n_past++;
                    n_session_consumed++;

                    if (n_session_consumed >= (int) session_tokens.size()) {
                        ++i;
                        break;
                    }
                }
                if (i > 0) {
                    embd.erase(embd.begin(), embd.begin() + i);
                }
            }

            for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
                int n_eval = (int) embd.size() - i;
                if (n_eval > params.n_batch) {
                    n_eval = params.n_batch;
                }
                
                result.append(string_from(ctx, embd));
                OH_LOG_ERROR(LOG_APP,"eval: %{public}s\n", string_from(ctx, embd).c_str());

                if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval))) {
                    OH_LOG_ERROR(LOG_APP,"%{public}s : failed to eval\n", __func__);
                    return 1;
                }

                n_past += n_eval;

                OH_LOG_ERROR(LOG_APP,"n_past = %{public}d\n", n_past);
                // Display total tokens alongside total time
                if (params.n_print > 0 && n_past % params.n_print == 0) {
                    OH_LOG_ERROR(LOG_APP,"\n\033[31mTokens consumed so far = %{public}d / %{public}d \033[0m\n", n_past, n_ctx);
                }
            }

            if (!embd.empty() && !path_session.empty()) {
                session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                n_session_consumed = session_tokens.size();
            }
        }

        embd.clear();

        if ((int) embd_inp.size() <= n_consumed && !is_interacting) {
            // optionally save the session on first sample (for faster prompt loading next time)
            if (!path_session.empty() && need_to_save_session && !params.prompt_cache_ro) {
                need_to_save_session = false;
                llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());

                OH_LOG_ERROR(LOG_APP,"saved session to %{public}s\n", path_session.c_str());
            }

            const llama_token id = common_sampler_sample(smpl, ctx, -1);

            common_sampler_accept(smpl, id, /* accept_grammar= */ true);

            // OH_LOG_ERROR(LOG_APP,"last: %{public}s\n", string_from(ctx, smpl->prev.to_vector()).c_str());

            embd.push_back(id);

            // echo this to console
            input_echo = true;

            // decrement remaining sampling budget
            --n_remain;

            OH_LOG_ERROR(LOG_APP,"n_remain: %{public}d\n", n_remain);
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            OH_LOG_ERROR(LOG_APP,"embd_inp.size(): %d, n_consumed: %d\n", (int) embd_inp.size(), n_consumed);
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);

                // push the prompt in the sampling context in order to apply repetition penalties later
                // for the prompt, we don't apply grammar rules
                common_sampler_accept(smpl, embd_inp[n_consumed], /* accept_grammar= */ false);

                ++n_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    break;
                }
            }
        }

        // display text
        if (input_echo && display) {
            for (auto id : embd) {
                const std::string token_str = common_token_to_piece(ctx, id, params.special);

                // Console/Stream Output
                LOG("%{public}s", token_str.c_str());

                // Record Displayed Tokens To Log
                // Note: Generated tokens are created one by one hence this check
                if (embd.size() > 1) {
                    // Incoming Requested Tokens
                    input_tokens.push_back(id);
                } else {
                    // Outgoing Generated Tokens
                    output_tokens.push_back(id);
                    output_ss << token_str;
                }
            }
        }

        // reset color to default if there is no pending user input
        if (input_echo && (int) embd_inp.size() == n_consumed) {
            console::set_display(console::reset);
            display = true;
        }

        // if not currently processing queued inputs;
        if ((int) embd_inp.size() <= n_consumed) {
            // check for reverse prompt in the last n_prev tokens
            if (!params.antiprompt.empty()) {
                const int n_prev = 32;
                const std::string last_output = common_sampler_prev_str(smpl, ctx, n_prev);

                is_antiprompt = false;
                // Check if each of the reverse prompts appears at the end of the output.
                // If we're not running interactively, the reverse prompt might be tokenized with some following characters
                // so we'll compensate for that by widening the search window a bit.
                for (std::string & antiprompt : params.antiprompt) {
                    size_t extra_padding = params.interactive ? 0 : 2;
                    size_t search_start_pos = last_output.length() > static_cast<size_t>(antiprompt.length() + extra_padding)
                        ? last_output.length() - static_cast<size_t>(antiprompt.length() + extra_padding)
                        : 0;

                    if (last_output.find(antiprompt, search_start_pos) != std::string::npos) {
                        if (params.interactive) {
                            is_interacting = true;
                        }
                        is_antiprompt = true;
                        break;
                    }
                }

                // check for reverse prompt using special tokens
                llama_token last_token = common_sampler_last(smpl);
                for (std::vector<llama_token> ids : antiprompt_ids) {
                    if (ids.size() == 1 && last_token == ids[0]) {
                        if (params.interactive) {
                            is_interacting = true;
                        }
                        is_antiprompt = true;
                        break;
                    }
                }

                if (is_antiprompt) {
                    OH_LOG_ERROR(LOG_APP,"found antiprompt: %{public}s\n", last_output.c_str());
                }
            }

            // deal with end of generation tokens in interactive mode
            if (llama_token_is_eog(model, common_sampler_last(smpl))) {
                OH_LOG_ERROR(LOG_APP,"found an EOG token\n");

                if (params.interactive) {
                    if (!params.antiprompt.empty()) {
                        // tokenize and inject first reverse prompt
                        const auto first_antiprompt = common_tokenize(ctx, params.antiprompt.front(), false, true);
                        embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
                        is_antiprompt = true;
                    }

                    if (params.enable_chat_template) {
                        chat_add_and_format(model, chat_msgs, "assistant", assistant_ss.str());
                    }
                    is_interacting = true;
                    LOG("\n");
                }
            }

            // if current token is not EOG, we add it to current assistant message
            if (params.conversation) {
                const auto id = common_sampler_last(smpl);
                assistant_ss << common_token_to_piece(ctx, id, false);
            }

            if (n_past > 0 && is_interacting) {
                OH_LOG_ERROR(LOG_APP,"waiting for user input\n");

                if (params.conversation) {
                    LOG("\n> ");
                }

                if (params.input_prefix_bos) {
                    OH_LOG_ERROR(LOG_APP,"adding input prefix BOS token\n");
                    embd_inp.push_back(llama_token_bos(model));
                }

                std::string buffer;
                if (!params.input_prefix.empty() && !params.conversation) {
                    OH_LOG_ERROR(LOG_APP,"appending input prefix: '%{public}s'\n", params.input_prefix.c_str());
                    LOG("%{public}s", params.input_prefix.c_str());
                }

                // color user input only
                console::set_display(console::user_input);
                display = params.display_prompt;

                std::string line;
                bool another_line = true;
                do {
                    another_line = console::readline(line, params.multiline_input);
                    buffer += line;
                } while (another_line);

                // done taking input, reset color
                console::set_display(console::reset);
                display = true;

                // Add tokens to embd only if the input buffer is non-empty
                // Entering a empty line lets the user pass control back
                if (buffer.length() > 1) {
                    // append input suffix if any
                    if (!params.input_suffix.empty() && !params.conversation) {
                        OH_LOG_ERROR(LOG_APP,"appending input suffix: '%{public}s'\n", params.input_suffix.c_str());
                        LOG("%{public}s", params.input_suffix.c_str());
                    }

                    OH_LOG_ERROR(LOG_APP,"buffer: '%{public}s'\n", buffer.c_str());

                    const size_t original_size = embd_inp.size();

                    if (params.escape) {
                        string_process_escapes(buffer);
                    }

                    bool format_chat = params.conversation && params.enable_chat_template;
                    std::string user_inp = format_chat
                        ? chat_add_and_format(model, chat_msgs, "user", std::move(buffer))
                        : std::move(buffer);
                    // TODO: one inconvenient of current chat template implementation is that we can't distinguish between user input and special tokens (prefix/postfix)
                    const auto line_pfx = common_tokenize(ctx, params.input_prefix, false, true);
                    const auto line_inp = common_tokenize(ctx, user_inp,            false, format_chat);
                    const auto line_sfx = common_tokenize(ctx, params.input_suffix, false, true);

                    OH_LOG_ERROR(LOG_APP,"input tokens: %{public}s\n", string_from(ctx, line_inp).c_str());

                    // if user stop generation mid-way, we must add EOT to finish model's last response
                    if (need_insert_eot && format_chat) {
                        llama_token eot = llama_token_eot(model);
                        embd_inp.push_back(eot == LLAMA_TOKEN_NULL ? llama_token_eos(model) : eot);
                        need_insert_eot = false;
                    }

                    embd_inp.insert(embd_inp.end(), line_pfx.begin(), line_pfx.end());
                    embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());
                    embd_inp.insert(embd_inp.end(), line_sfx.begin(), line_sfx.end());

                    for (size_t i = original_size; i < embd_inp.size(); ++i) {
                        const llama_token token = embd_inp[i];
                        output_tokens.push_back(token);
                        output_ss << common_token_to_piece(ctx, token);
                    }

                    // reset assistant message
                    assistant_ss.str("");

                    n_remain -= line_inp.size();
                    OH_LOG_ERROR(LOG_APP,"n_remain: %{public}d\n", n_remain);
                } else {
                    OH_LOG_ERROR(LOG_APP,"empty line, passing control back\n");
                }

                input_echo = false; // do not echo this again
            }

            if (n_past > 0) {
                if (is_interacting) {
                    common_sampler_reset(smpl);
                }
                is_interacting = false;
            }
        }

        // end of generation
        if (!embd.empty() && llama_token_is_eog(model, embd.back()) && !(params.interactive)) {
            LOG(" [end of text]\n");
            break;
        }

        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        // We skip this logic when n_predict == -1 (infinite) or -2 (stop at context size).
        if (params.interactive && n_remain <= 0 && params.n_predict >= 0) {
            n_remain = params.n_predict;
            is_interacting = true;
        }
    }

    if (!path_session.empty() && params.prompt_cache_all && !params.prompt_cache_ro) {
        LOG("\n%{public}s: saving final output to session file '%{public}s'\n", __func__, path_session.c_str());
        llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
    }

    LOG("\n\n");
    common_perf_print(ctx, smpl);

    common_sampler_free(smpl);

    llama_backend_free();

    ggml_threadpool_free_fn(threadpool);
    ggml_threadpool_free_fn(threadpool_batch);

    return 0;
}

static napi_value Add(napi_env env, napi_callback_info info)
{
    size_t argc = 2;
    napi_value args[2] = {nullptr};

    napi_get_cb_info(env, info, &argc, args , nullptr, nullptr);

    napi_valuetype valuetype0;
    napi_typeof(env, args[0], &valuetype0);

    napi_valuetype valuetype1;
    napi_typeof(env, args[1], &valuetype1);

    double value0;
    napi_get_value_double(env, args[0], &value0);

    double value1;
    napi_get_value_double(env, args[1], &value1);

    napi_value sum;
    napi_create_double(env, value0 + value1, &sum);

    return sum;

}

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cstring>

// 函数：将参数字符串转换为 argc 和 argv
void parseCommandLine(const std::string& input, int& argc, char**& argv) {
    // 用于存储分割后的参数
    std::vector<std::string> args;
    std::string currentArg;
    bool inQuotes = false; // 是否在引号内

    for (size_t i = 0; i < input.size(); ++i) {
        char c = input[i];

        if (c == '\"') {
            // 如果遇到引号，切换引号状态
            inQuotes = !inQuotes;
        } else if (c == ' ' && !inQuotes) {
            // 如果遇到空格且不在引号内，将当前参数保存
            if (!currentArg.empty()) {
                args.push_back(currentArg);
                currentArg.clear();
            }
        } else {
            // 将字符加入当前参数
            currentArg += c;
        }
    }

    // 保存最后一个参数（如果存在）
    if (!currentArg.empty()) {
        args.push_back(currentArg);
    }

    // 设置 argc
    argc = static_cast<int>(args.size());

    // 分配 argv
    argv = new char*[argc];
    for (int i = 0; i < argc; ++i) {
        argv[i] = new char[args[i].size() + 1];
        std::strcpy(argv[i], args[i].c_str());
    }
}


std::string global_result;
static napi_value Openllama(napi_env env, napi_callback_info info)
{
     //输入参数个数

    size_t argc = 1;

    //输入参数数组

    napi_value args[1] = {nullptr};

    //将获取的传入参数放入数组中

    if (napi_ok != napi_get_cb_info(env, info, &argc, args, nullptr, nullptr)) {

        return nullptr;

    }

//上面的你就能把输入的字符串放入args[0]中了，下面就是写对应的调用逻辑

//记录长度

    size_t typeLen = 0;

    //char类型的转换

    char *str = nullptr;

    //写入缓存，获得args[0]对应的char长度

    napi_get_value_string_utf8(env, args[0], nullptr, 0, &typeLen);

    //napi_get_value_string_utf8（env，数组对象，char，缓存长度，获取的长度）主要作用是通过缓存复制的方法，将对象转换为char，复制到缓存中，获取长度

    str = new char[typeLen + 1];

   //获取输入的字符串转换为char类型的str

    napi_get_value_string_utf8(env, args[0], str, typeLen + 1, &typeLen);

    //然后你就可以写对应的加密之类的操作了，这个自己写，我跳过了
       // 示例输入
    std::string input(str, typeLen);

    int _argc;
    char** _argv;

    // 解析输入
    parseCommandLine(input, _argc, _argv);
    std::cout<<"argc: "<<_argc<<std::endl;
    OH_LOG_ERROR(LOG_APP, "Argc:%{public}d, argv:%{public}s", _argc,str);
    // 输出 argc 和 argv
    std::future<void> async_result = std::async(std::launch::async, [&]() {
        main_function(_argc, _argv, global_result);
    });

    
    // 释放动态分配的内存
//     for (int i = 0; i < argc; ++i) {
//         delete[] argv[i];
//     }
//     delete[] argv;
//
//     main_function(, char **argv)
    
    //创建输出对象

    napi_value output;

    //将char类型的str，赋值给output,类型为string

    napi_create_string_utf8(env, global_result.c_str(), global_result.size() , &output);

    //返回的是长度

    //napi_create_double(env, typeLen, &output);

    return output;

//    return args[0];//这个是直接返回输入对象

}

static napi_value Getllama(napi_env env, napi_callback_info info)
{
     //输入参数个数

    size_t argc = 1;

    //输入参数数组

    napi_value args[1] = {nullptr};

    //将获取的传入参数放入数组中

    if (napi_ok != napi_get_cb_info(env, info, &argc, args, nullptr, nullptr)) {

        return nullptr;

    }

//上面的你就能把输入的字符串放入args[0]中了，下面就是写对应的调用逻辑

//记录长度

    size_t typeLen = 0;

    //char类型的转换

    char *str = nullptr;

    //写入缓存，获得args[0]对应的char长度

    napi_get_value_string_utf8(env, args[0], nullptr, 0, &typeLen);

    //napi_get_value_string_utf8（env，数组对象，char，缓存长度，获取的长度）主要作用是通过缓存复制的方法，将对象转换为char，复制到缓存中，获取长度

    str = new char[typeLen + 1];

   //获取输入的字符串转换为char类型的str

    napi_get_value_string_utf8(env, args[0], str, typeLen + 1, &typeLen);

    //然后你就可以写对应的加密之类的操作了，这个自己写，我跳过了
       // 示例输入
    std::string input(str, typeLen);

    int _argc;
    char** _argv;

    // 解析输入
//     parseCommandLine(input, _argc, _argv);
//     std::cout<<"argc: "<<_argc<<std::endl;
//     OH_LOG_ERROR(LOG_APP, "Argc:%{public}d, argv:%{public}s", _argc,str);
    // 输出 argc 和 argv
//     main_function(_argc,_argv);

    
    // 释放动态分配的内存
//     for (int i = 0; i < argc; ++i) {
//         delete[] argv[i];
//     }
//     delete[] argv;
//
//     main_function(, char **argv)
    
    //创建输出对象

    napi_value output;

    //将char类型的str，赋值给output,类型为string

    napi_create_string_utf8(env, global_result.c_str(), global_result.size() , &output);

    //返回的是长度

    //napi_create_double(env, typeLen, &output);

    return output;

//    return args[0];//这个是直接返回输入对象

}
EXTERN_C_START
static napi_value Init(napi_env env, napi_value exports)
{
    napi_property_descriptor desc[] = {
        { "add", nullptr, Add, nullptr, nullptr, nullptr, napi_default, nullptr },
        { "openllama", nullptr, Openllama, nullptr, nullptr, nullptr, napi_default, nullptr },
        { "getllama", nullptr, Getllama, nullptr, nullptr, nullptr, napi_default, nullptr },
    };
    napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]), desc);
    return exports;
}
EXTERN_C_END

static napi_module demoModule = {
    .nm_version = 1,
    .nm_flags = 0,
    .nm_filename = nullptr,
    .nm_register_func = Init,
    .nm_modname = "entry",
    .nm_priv = ((void*)0),
    .reserved = { 0 },
};

extern "C" __attribute__((constructor)) void RegisterEntryModule(void)
{
    napi_module_register(&demoModule);
}
