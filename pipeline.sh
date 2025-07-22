#!/bin/bash

set -e

# 从环境变量获取配置
API_KEY="${OPENAI_API_KEY}"
API_BASE="${OPENAI_API_BASE}"
MODEL="${OPENAI_MODEL:-gpt-4o-mini}"

# 检查必需的环境变量
if [ -z "$API_KEY" ]; then
    echo "错误: 请设置环境变量 OPENAI_API_KEY"
    exit 1
fi

if [ -z "$API_BASE" ]; then
    echo "错误: 请设置环境变量 OPENAI_API_BASE"
    exit 1
fi

echo "========================================"
echo "🚀 LPA Pipeline 开始运行"
echo "时间: $(date)"
echo "使用模型: $MODEL"
echo "========================================"

echo "📝 生成完整数据."
python seed_generator.py --mode complete --output_dir data/generated --num_personas 5 --scenes_per_persona 16 --filename complete_dataset.json
echo "✅ 完整数据集生成完成"

echo "🔍 生成query"
python query_generator.py --persona_db data/generated/complete_dataset.json --output_dir test_query_output --format json
echo "✅ 查询数据生成完成"


echo ""
echo "🧪 Learning"
python unified_main.py \
    --persona test_query_output/persona_005.json \
    --persona_db test_query_output/persona_005_db.json\
    --openai_api_key "$API_KEY" \
    --openai_api_base "$API_BASE" \
    --model_name "$MODEL" \
    --update_frequency 3 \
    --max_retries 3 \
    --log_level INFO \
    --log_dir "logs_pipeline" \
    --db_prefix "person_conversations_learning_0721"

echo ""
echo "========================================"
echo "🎉 Pipeline执行完成!"
echo "生成的文件:"
echo "  - 测试数据: test_data/"
echo "  - 查询数据: test_query_output/"
echo "  - 日志文件: logs_pipeline/"
echo "  - 对话数据: data/pipeline_conversations/"
echo "========================================" 