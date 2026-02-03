from cs336_basics.train_bpe import train_bpe_streaming_read,train_bpe_restart

import time
import resource
import sys
#"/home/jerry/Downloads/owt_train.txt"
#"l.txt",
def main():
    train_bpe_streaming_read(
        "/home/jerry/Downloads/owt_train.txt",
        32000,
        [],
        use_pre=False
    )
    #train_bpe_restart(
    #    "trained_values/openwebtext_success/vocab_final.pkl",
    #    "trained_values/openwebtext_success/merges_final.pkl",
    #    "trained_values/openwebtext_success/l.txt",
    #    ["<|endoftext|>"],
    #    32000
    #)

if __name__ == "__main__":
    # 记录开始时间
    start_time = time.time()
    
    # 记录开始内存
    start_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    
    # 运行主程序
    main()
    
    # 记录结束时间和内存
    end_time = time.time()
    end_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    
    # 计算并打印结果
    print(f"执行时间: {end_time - start_time:.4f} 秒")
    print(f"最大内存使用: {(end_memory - start_memory) / 1024:.2f} MB (Linux)")  # Linux上单位是KB
