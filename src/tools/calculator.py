import ast
import operator
from typing import Dict, Any

# Whitelist of safe operators — avoids exec/eval security pitfalls
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class UnsafeExpressionError(ValueError):
    pass


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluates an AST node using only whitelisted operators."""
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise UnsafeExpressionError(f"Non-numeric constant: {node.value}")
        return node.value
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise UnsafeExpressionError(f"Operator not allowed: {op_type}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return SAFE_OPERATORS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise UnsafeExpressionError(f"Unary operator not allowed: {op_type}")
        return SAFE_OPERATORS[op_type](_safe_eval(node.operand))
    else:
        raise UnsafeExpressionError(f"Unsupported AST node type: {type(node)}")


def calculator(expression: str) -> str:
    """
    Evaluates a math expression string using a whitelist-only AST evaluator.
    Returns the numeric result as a string, or an error message.

    Args:
        expression: A string like "12 * (3 + 4) / 2"
    Returns:
        String result, e.g. "42" or "Error: ..."
    """
    expression = expression.strip()
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        # Return integer string if result is a whole number
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero"
    except UnsafeExpressionError as e:
        return f"Error: Unsafe expression — {e}"
    except SyntaxError as e:
        return f"Error: Invalid syntax — {e}"
    except Exception as e:
        return f"Error: {e}"


# Tool descriptor consumed by ReActAgent.__init__
CALCULATOR_TOOL: Dict[str, Any] = {
    "name": "calculator",
    "description": (
        "Evaluates a mathematical expression and returns the numeric result. "
        "Supports +, -, *, /, //, %, ** and parentheses. "
        "Input must be a valid arithmetic expression string, e.g. '(3 + 5) * 12 / 4'. "
        "Do NOT include variable names, units, or words — only numbers and operators."
    ),
    "fn": calculator,
}
