#!/bin/bash

# --- 配置部分 ---
# 1. 定义日志目录
LOG_DIR="./task_logs/game_of_24"

# 2. 定义输出文件的前缀
LOG_PREFIX="training_output_"

# 3. 定义单个日志文件的最大大小 (1MB)
LOG_SIZE='1M'

# 4. 定义最多保留的日志文件数量 (实现 backupCount 功能)
MAX_LOG_FILES=5000

# 5. 您要运行的 Python 命令 (用引号括起来)
PYTHON_COMMAND="python examples/game_of_24/game24_octo.py"
# 或者更复杂的命令，例如：
# PYTHON_COMMAND="python -m agentlightning.verl algorithm.adv_estimator=grpo data.train_batch_size=8"

rm -rf $LOG_DIR
mkdir $LOG_DIR
# 2. 执行 Python 命令，并将输出（stdout 和 stderr）通过管道传递给 split 进行分割
#    split 会生成 training_output_00, training_output_01, ... 等文件
echo "Starting the task..."

$PYTHON_COMMAND 2>&1 | split -b "$LOG_SIZE" -d - "$LOG_DIR/$LOG_PREFIX"

# 3. 获取 split 命令的退出状态
SPLIT_EXIT_CODE=${PIPESTATUS[1]} # PIPESTATUS[1] 是 split 的退出码

# 4. 检查命令是否成功执行
if [ $SPLIT_EXIT_CODE -eq 0 ]; then
    echo "Task completed successfully."
else
    echo "Error: The task or log splitting failed with exit code $SPLIT_EXIT_CODE."
    exit $SPLIT_EXIT_CODE
fi

# 5. 清理旧的日志文件，只保留最新的 $MAX_LOG_FILES 个
#    按文件名降序排序 (00, 01, 02... -> 最新的在最前面)，跳过前 $MAX_LOG_FILES 个，删除其余的
echo "Cleaning up old log files, keeping the latest $MAX_LOG_FILES..."
ls -1 "$LOG_DIR"/"$LOG_PREFIX"* 2>/dev/null | sort -r | tail -n +$((MAX_LOG_FILES + 1)) | xargs rm -f

echo "Log files are saved in: $LOG_DIR"