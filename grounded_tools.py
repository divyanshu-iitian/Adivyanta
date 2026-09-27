"""Small exact-answer helpers for the chat app, separate from model generation."""
import ast
from fractions import Fraction
import re


EXPRESSION = re.compile(
    r"^\s*(?:(?:what is|calculate|compute|solve|kitna hai|batao)\s+)?"
    r"([0-9\s+*/().-]{3,80})\s*"
    r"(?:(?:kitna hota hai|kitna hai|का उत्तर क्या है|का जवाब क्या है|कितना है)\s*)?"
    r"(?:\?|=|।)?\s*$", re.I
)


def _value(node):
    if isinstance(node, ast.Expression):
        return _value(node.body)
    if isinstance(node, ast.Constant) and type(node.value) is int and abs(node.value) <= 1_000_000:
        return Fraction(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _value(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left, right = _value(node.left), _value(node.right)
        if isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        elif isinstance(node.op, ast.Mult):
            result = left * right
        else:
            if not right:
                raise ZeroDivisionError
            result = left / right
        if abs(result.numerator) > 10**12 or result.denominator > 10**12:
            raise ValueError("result too large")
        return result
    raise ValueError("unsupported expression")


def arithmetic_answer(message):
    match = EXPRESSION.fullmatch(message)
    if not match:
        return None
    expression = match.group(1).strip()
    if not re.search(r"[+*/-]", expression):
        return None
    try:
        answer = _value(ast.parse(expression, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, RecursionError):
        return "I can't calculate that expression."
    return str(answer.numerator) if answer.denominator == 1 else f"{answer.numerator}/{answer.denominator}"
