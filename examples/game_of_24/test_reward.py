import re
import ast
from functools import wraps
import builtins  # 用于调用原始 eval

# 定义 reward 装饰器
def reward(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return float(result) if result in (0.0, 1.0) else 0.0
        except Exception as e:
            print(f"[Reward] Error: {e}")
            return 0.0
    return wrapper

@reward
def eval(groundtruth, answer_extracted) -> float:
    """
    奖励函数：判断 answer_extracted 是否是 24 点的有效解
    - 函数名必须是 eval
    - 不直接使用 eval()，而是用 builtins.eval 避免冲突
    """
    # Step 1: 从 groundtruth 中提取目标数字
    def extract_numbers(gt_list):
        for expr in gt_list:
            # 统一符号
            expr = expr.replace('\u00d7', '*').replace('×', '*').replace('−', '-').replace(' ', '')
            nums = re.findall(r'\d+', expr)
            if len(nums) == 4:
                return sorted(int(x) for x in nums)
        return None

    target_nums = extract_numbers(groundtruth)
    if target_nums is None:
        return 0.0

    # Step 2: 处理 answer_extracted
    expr = str(answer_extracted).strip()
    if not expr:
        return 0.0

    # 统一运算符
    expr_clean = expr.replace('\u00d7', '*').replace('×', '*').replace('−', '-').replace(' ', '')

    # Step 3: 提取使用的数字
    used_strs = re.findall(r'\d+', expr_clean)
    if len(used_strs) != 4:
        return 0.0
    try:
        used_nums = sorted(int(x) for x in used_strs)
    except:
        return 0.0

    if used_nums != target_nums:
        return 0.0  # 数字不匹配

    # Step 4: 检查合法字符
    if not re.fullmatch(r'[\d\+\-\*\/\(\)]+', expr_clean):
        return 0.0

    # Step 5: 使用 builtins.eval 安全计算（避免与函数名 eval 冲突）
    try:
        ast.parse(expr_clean, mode='eval')  # 语法检查
        result = builtins.eval(expr_clean)  # ✅ 使用 builtins.eval
        return 1.0 if abs(result - 24.0) < 1e-5 else 0.0
    except:
        return 0.0


if __name__ == "__main__":
    groundtruth = [
        "7\u00d74-3-1",
        "(3+1)\u00d77-4",
        "7\u00d73+4-1",
        "(4-1)\u00d77+3"
    ]

    test_cases = [
        "(3+1)*7-4",      # 正确：4*7-4=24
        "(3+1)×7-4",      # 正确：含 ×
        "7×4-3-1",        # 正确：28-3-1=24
        "(7-1)*(4+3)",    # 错误：6*7=42
        "7+4+3+1",        # 错误：15
        "7+8+2+9-1",        # 错误：15
        "7+8+0+9",        # 错误：15
    ]

    for ans in test_cases:
        score = eval(groundtruth, ans)  # 这里调用的是你的 eval 函数
        print(f"Answer: {ans} → Score: {score}")