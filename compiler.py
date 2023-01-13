# This is the Compiler that is made by Ritik Jaiswal
# Import the string and math modules in Python
# High Level Language Phase

import string
import math

# Declare the DIGITS, WORDS and ALPHABETS_DIGITS

DIGITS = '0123456789'
WORDS = string.ascii_letters
ALPHABETS_DIGITS = WORDS + DIGITS

# I have declared the Position class and here the first phase (Lexical Analysis) execution is started of compiler

class Position_RitikJaiswal:
  def __init__(self, index, ln, column, fn, ftxt):

    #It contains index, ln, column, fn, ftxt

    self.index = index
    self.ln = ln
    self.column = column
    self.fn = fn
    self.ftxt = ftxt

  def advance(self, current_char=None):
    self.index += 1
    self.column += 1

    if current_char == '\n':
      self.ln += 1
      self.column = 0

    return self

  def copy(self):
    return Position_RitikJaiswal(self.index, self.ln, self.column, self.fn, self.ftxt)

# High Level Language Phase

# These are the several types tokens of that I have used to make this compiler like ADD, SUB, MUL, DIV and etc. 

Compiler_Int = 'Integer'
Compiler_Float = 'Float'
Compiler_String = 'String'
Compiler_Identifier = 'Identifier'
Compiler_Keyword = 'Keyword'
Compiler_Add = 'Addition'
Compiler_Subtract = 'Subtraction'
Compiler_Multiply = 'Multiplication'
Compiler_Divide = 'Division'
Compiler_Assignment = 'Assignment'
Compiler_LeftParenthesis = 'Left Parenthesis'
Compiler_RightParenthesis = 'Right Parenthesis '
Compiler_Equal	= 'Eqaul'
Compiler_LessThan	= 'Less Than'
Compiler_GreaterThan = 'Greater Than'
Compiler_Comma = 'Comma'
Compiler_NewLine = 'New Line'
Compiler_EndOfLine = 'End of Line'

# Declare Keywords

KEYWORDS = ['VAR','AND','OR','NOT','THEN','FUN','IF','FOR','WHILE','RETURN']

# Phase-1: Lexical Analysis: Implementation of Lexical Analysis and here the integration of tokens happens
# Create a class for Tokens and perform the following operation

class Tokens_RitikJaiswal:
  def __init__(self, type_, value=None, starting_position=None, ending_position=None):
    self.type = type_
    self.value = value
     #if condition for starting_position
    if starting_position:
      self.starting_position = starting_position.copy()
      self.ending_position = starting_position.copy()
      self.ending_position.advance()
    #if condition for ending_position
    if ending_position:
      self.ending_position = ending_position.copy()

  def matches(self, type_, value):
    return self.type == type_ and self.value == value
  
  def __repr__(self):
    if self.value: return f'{self.type}:{self.value}'
    return f'{self.type}'
# I have created an Error class where all the phases of errors are declared and executed

class ErrorHandle_RitikJaiswal:
  def __init__(self, starting_position, ending_position, error_name, details):
    self.starting_position = starting_position
    self.ending_position = ending_position
    self.error_name = error_name
    self.details = details
  
  def as_string(self):
    result  = f'{self.error_name}: {self.details}\n'
    result += f'File {self.starting_position.fn}, line {self.starting_position.ln + 1}'
    
    return result

# These are the following Errors which are called in Compiler: Illegal Character, Expected Character, Invalid Syntax, Runtime Error

class IllegalCharError_RitikJaiswal(ErrorHandle_RitikJaiswal):
  def __init__(self, starting_position, ending_position, details):
    super().__init__(starting_position, ending_position, 'Illegal Character', details)

class ExpectedCharError_RitikJaiswal(ErrorHandle_RitikJaiswal):
  def __init__(self, starting_position, ending_position, details):
    super().__init__(starting_position, ending_position, 'Expected Character', details)

class InvalidSyntaxError_RitikJaiswal(ErrorHandle_RitikJaiswal):
  def __init__(self, starting_position, ending_position, details=''):
    super().__init__(starting_position, ending_position, 'Invalid Syntax', details)

class RunTimeError_RitikJaiswal(ErrorHandle_RitikJaiswal):
  def __init__(self, starting_position, ending_position, details, context):
    super().__init__(starting_position, ending_position, 'Runtime Error', details)
    self.context = context

  def as_string(self):
    result  = self.develop_traceback()
    result += f'{self.error_name}: {self.details}'
    return result

  def develop_traceback(self):
    result = ''
    pos = self.starting_position
    ctx = self.context
    return 'Traceback (most recent call last):\n' + result

#Phase-1: Lexical Analysis: Implementation of Lexical Analysis and here the integration of tokens happens

class Lexical_Analysis:
  def __init__(self, fn, text):
    self.fn = fn
    self.text = text
    self.position = Position_RitikJaiswal(-1, 0, -1, fn, text)
    self.current_char = None
    self.advance()
  
  def advance(self):
    self.position.advance(self.current_char)
    self.current_char = self.text[self.position.index] if self.position.index < len(self.text) else None

#Definition of all tokens and Define the make_Tokenss function

  def make_Tokenss(self):
    Tokenss = []
    while self.current_char != None:
      if self.current_char in ' \t':
        self.advance()
      elif self.current_char == '#':         #COMMENT
        self.skip_comment()
      elif self.current_char in ';\n':
        Tokenss.append(Tokens_RitikJaiswal(Compiler_NewLine, starting_position=self.position))
      elif self.current_char in DIGITS:
        Tokenss.append(self.make_number())
      elif self.current_char in WORDS:
        Tokenss.append(self.make_identifier())
      elif self.current_char == '"':                     #STRING
        Tokenss.append(self.make_string())
      elif self.current_char == '+':            #ADDITION
        Tokenss.append(Tokens_RitikJaiswal(Compiler_Add, starting_position=self.position))
        self.advance()
      elif self.current_char == '-':           #SUBTRACTION
        Tokenss.append(Tokens_RitikJaiswal(Compiler_Subtract, starting_position=self.position))
        self.advance()
      elif self.current_char == '*':           #MULTIPLICATION
        Tokenss.append(Tokens_RitikJaiswal(Compiler_Multiply, starting_position=self.position))
        self.advance()
      elif self.current_char == '/':          #DIVISION
        Tokenss.append(Tokens_RitikJaiswal(Compiler_Divide, starting_position=self.position))
        self.advance()
      elif self.current_char == '(':        #LEFT PARENTHESIS
        Tokenss.append(Tokens_RitikJaiswal(Compiler_LeftParenthesis, starting_position=self.position))
        self.advance()
      elif self.current_char == ')':         #RIGHT PARENTHESIS
        Tokenss.append(Tokens_RitikJaiswal(Compiler_RightParenthesis, starting_position=self.position))
        self.advance()
      elif self.current_char == '=':         #EQUALS
        Tokenss.append(self.make_equals())
      elif self.current_char == '<':
        Tokenss.append(self.make_less_than())    #LESS THAN
      elif self.current_char == '>':
        Tokenss.append(self.make_greater_than())      #GREATER THAN
      elif self.current_char == ',':
        Tokenss.append(Tokens_RitikJaiswal(Compiler_Comma, starting_position=self.position))
        self.advance()
      else:
        starting_position = self.position.copy()
        char = self.current_char
        self.advance()
        return [], IllegalCharError_RitikJaiswal(starting_position, self.position, "'" + char + "'")      #ILLEGAL CHARACTER
    Tokenss.append(Tokens_RitikJaiswal(Compiler_EndOfLine, starting_position=self.position))
    return Tokenss, None
    
#Define the make_number function
  def make_number(self):
    num_str = ''
    dot_count = 0
    starting_position = self.position.copy()

    while self.current_char != None and self.current_char in DIGITS + '.':
      if self.current_char == '.':
        if dot_count == 1: break
        dot_count += 1
      num_str += self.current_char
      self.advance()

# Foe INT and FLOAT values
    if dot_count == 0:
      return Tokens_RitikJaiswal(Compiler_Int, int(num_str), starting_position, self.position)
    else:
      return Tokens_RitikJaiswal(Compiler_Float, float(num_str), starting_position, self.pos)

  def make_identifier(self):
    id_str = ''
    starting_position = self.position.copy()

    while self.current_char != None and self.current_char in ALPHABETS_DIGITS + '_':
      id_str += self.current_char
      self.advance()

    tok_type = Compiler_Keyword if id_str in KEYWORDS else Compiler_Identifier
    return Tokens_RitikJaiswal(tok_type, id_str, starting_position, self.position)

     #Make Eqaul Function  

  def make_equals(self):
    tok_type = Compiler_Assignment
    pos_start = self.pos.copy()
    self.advance()

    if self.current_char == '=':
      self.advance()
      tok_type = Compiler_Equal

    return Tokens_RitikJaiswal(tok_type, pos_start=pos_start, pos_end=self.pos)

    # Skip Comments Function

  def skip_comment(self):
    self.advance()

    while self.current_char != '\n':
      self.advance()

    self.advance()
  
# Phase-2: Syntax Analysis: Syntax Analysis is the second phase of compiler which is also called as parsing

# Create a Node class and create the subclasses of respective node

class NumberNode_RitikJaiswal:
  def __init__(self, tok):
    self.tok = tok

    self.pos_start = self.tok.pos_start
    self.pos_end = self.tok.pos_end

  def __repr__(self):
    return f'{self.tok}'

# List Node of Compiler

class ListNode_RitikJaiswal:
  def __init__(self, element_nodes, pos_start, pos_end):
    self.element_nodes = element_nodes
    self.pos_start = pos_start
    self.pos_end = pos_end

# VarAccess Node of Compiler

class VarAccessNode_RitikJaiswal:
  def __init__(self, var_name_token):
    self.var_name_token = var_name_token
    self.pos_start = self.var_name_token.pos_start
    self.pos_end = self.var_name_token.pos_end

# VarAssign Node of Compiler

class VarAssignNode_RitikJaiswal:
  def __init__(self, var_name_token, value_node):
    self.var_name_token = var_name_token
    self.value_node = value_node
    self.pos_start = self.var_name_token.pos_start
    self.pos_end = self.value_node.pos_end

# BinOp Node of Compiler

class BinOpNode_RitikJaiswal:
  def __init__(self, left_node, operator_token, right_node):
    self.left_node = left_node
    self.operator_token = operator_token
    self.right_node = right_node
    self.pos_start = self.left_node.pos_start
    self.pos_end = self.right_node.pos_end
  def __repr__(self):
    return f'({self.left_node}, {self.operator_token}, {self.right_node})'

# UnayOp Node of Compiler

class UnaryOpNode_RitikJaiswal:
  def __init__(self, operator_token, node):
    self.operator_token = operator_token
    self.node = node
    self.pos_start = self.operator_token.pos_start
    self.pos_end = node.pos_end
  def __repr__(self):
    return f'({self.operator_token}, {self.node})'

# If Node of Compiler

class IfNode_RitikJaiswal:
  def __init__(self, cases, else_case):
    self.cases = cases
    self.else_case = else_case
    self.pos_start = self.cases[0][0].pos_start
    self.pos_end = (self.else_case or self.cases[len(self.cases) - 1])[0].pos_end

# For Node of Compiler

class ForNode_RitikJaiswal:
  def __init__(self, var_name_token, start_value_node, end_value_node, step_value_node, body_node, should_return_null):
    self.var_name_token = var_name_token
    self.start_value_node = start_value_node
    self.end_value_node = end_value_node
    self.step_value_node = step_value_node
    self.body_node = body_node
    self.should_return_null = should_return_null

# While Node of Compiler

class WhileNode_RitikJaiswal:
  def __init__(self, condition_node, body_node, should_return_null):
    self.condition_node = condition_node
    self.body_node = body_node
    self.should_return_null = should_return_null
    self.pos_start = self.condition_node.pos_start
    self.pos_end = self.body_node.pos_end

# Call Node of Compiler

class CallNode_RitikJaiswal:
  def __init__(self, node_to_call, arg_nodes):
    self.node_to_call = node_to_call
    self.arg_nodes = arg_nodes
    self.starting_position = self.node_to_call.starting_position
    if len(self.arg_nodes) > 0:
      self.ending_position = self.arg_nodes[len(self.arg_nodes) - 1].ending_position
    else:
      self.ending_position = self.node_to_call.ending_position

# Return Node of Compiler

class ReturnNode_RitikJaiswal:
  def __init__(self, node_to_return, starting_position, ending_position):
    self.node_to_return = node_to_return
    self.starting_position = starting_position
    self.ending_position = ending_position

# This is the Parser of my Compiler and here is the whole procedure for Parser

class Parser:
  def __init__(self, Tokenss):
    self.Tokenss = Tokenss
    self.tok_index = -1
    self.advance()
#Advance Function
  def advance(self):             
    self.tok_index += 1
    self.update_current_tok()
    return self.current_tok
#Reverse Function
  def reverse(self, amount=1):    
    self.tok_index -= amount
    self.update_current_tok()
    return self.current_tok
#Update Function
  def update_current_tok(self):
    if self.tok_index >= 0 and self.tok_index < len(self.Tokenss):
      self.current_tok = self.Tokenss[self.tok_index]
#Parse Function
  def parse(self):
    res = self.statements()
    if not res.error and self.current_tok.type != Compiler_EndOfLine:
      return res.failure(InvalidSyntaxError_RitikJaiswal(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Token cannot appear after previous tokens"
      ))
    return 

  def statements(self):
    res = Parse_Outcome()
    statements = []
    starting_position = self.current_tok.starting_position.copy()
    statement = res.register(self.statement())
    if res.error: return res
    statements.append(statement)
    more_statements = True

  def arith_expr(self):
    return self.bin_op(self.term, (Compiler_Add, Compiler_Subtract))
  def term(self):
    return self.bin_op(self.factor, (Compiler_Multiply, Compiler_Divide))
  def factor(self):
    res = Parse_Outcome()
    tok = self.current_tok
    return self.power()
  def list_expr(self):
    res = Parse_Outcome()
    element_nodes = []
    starting_position = self.current_tok.starting_position.copy()
  def if_expr_cases(self, case_keyword):
    res = Parse_Outcome()
    cases = []
    else_case = None

    if not self.current_tok.matches(Compiler_Keyword, case_keyword):
      return res.failure(InvalidSyntaxError_RitikJaiswal(
        self.current_tok.starting_position, self.current_tok.ending_position,
        f"Expected '{case_keyword}'"
      ))

    res.register_advancement()
    self.advance()
    condition = res.register(self.expr())
    if res.error: return res

    if not self.current_tok.matches(Compiler_Keyword, 'THEN'):
      return res.failure(InvalidSyntaxError_RitikJaiswal(
        self.current_tok.starting_position, self.current_tok.ending_position,
        f"Expected 'then'"
      ))

    res.register_advancement()
    self.advance()

# Declare a Value class

class Value:
  def __init__(self):
    self.set_pos()
    self.set_context()

  def set_pos(self, starting_position=None, ending_position=None):
    self.starting_position = starting_position
    self.ending_position = ending_position
    return self

  def set_context(self, context=None):
    self.context = context
    return self

  def added_to(self, other):
    return None, self.illegal_operation(other)

  def subtracted_by(self, other):
    return None, self.illegal_operation(other)

  def multiplied_by(self, other):
    return None, self.illegal_operation(other)

  def divided_by(self, other):
    return None, self.illegal_operation(other)
    
class Number(Value):
  def __init__(self, value):
    super().__init__()
    self.value = value

  def added_to(self, other):
    if isinstance(other, Number):
      return Number(self.value + other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def subtracted_by(self, other):
    if isinstance(other, Number):
      return Number(self.value - other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def multiplied_by(self, other):
    if isinstance(other, Number):
      return Number(self.value * other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def dived_by(self, other):
    if isinstance(other, Number):
      if other.value == 0:
        return None, RunTimeError_RitikJaiswal(
          other.starting_position, other.ending_position,
          'Division by zero',
          self.context
        )

      return Number(self.value / other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def copy(self):
    copy = Number(self.value)
    copy.set_pos(self.starting_position, self.ending_position)
    copy.set_context(self.context)
    return copy

  def is_true(self):
    return self.value != 0

  def __str__(self):
    return str(self.value)
  
  def __repr__(self):
    return str(self.value)

Number.null = Number(0)
Number.false = Number(0)
Number.true = Number(1)
Number.math_PI = Number(math.pi)

# Phase-3: Semantic Analysis: Semantic Analysis is the third phase of compiler

class String(Value):
  def __init__(self, value):
    super().__init__()
    self.value = value

  def added_to(self, other):
    if isinstance(other, String):
      return String(self.value + other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def multiplied_by(self, other):
    if isinstance(other, Number):
      return String(self.value * other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)
  def is_true(self):
    return len(self.value) > 0
  def copy(self):
    copy = String(self.value)
    copy.set_pos(self.starting_position, self.ending_position)
    copy.set_context(self.context)
    return copy

# Base Function
class BaseFunction(Value):
  def __init__(self, name):
    super().__init__()
    self.name = name or "<Anonymous>"
  def generate_new_context(self):
    new_context = Context(self.name, self.context, self.starting_position)
    return new_context
  def populate_args(self, argument_names, args, exec_ctx):
    for i in range(len(args)):
      argument_name = argument_names[i]
      argument_value = args[i]
      argument_value.set_context(exec_ctx)
      exec_ctx.symbol_table.set(argument_name, argument_value)

#Create a function class and call the body_node, argument_names and should_auto_return
class Function(BaseFunction):
  def __init__(self, name, body_node, argument_names, should_auto_return):
    super().__init__(name)
    self.body_node = body_node
    self.argument_names = argument_names
    self.should_auto_return = should_auto_return
    interpreter = Interpreter()
    exec_ctx = self.generate_new_context()

  def copy(self):
    copy = Function(self.name, self.body_node, self.argument_names, self.should_auto_return)
    copy.set_context(self.context)
    copy.set_pos(self.starting_position, self.ending_position)
    return copy
  def __repr__(self):
    return f"<function {self.name}>"

#Here BuildInFunction is executed to call the several functions
class BuiltInFunction(BaseFunction):
  def __init__(self, name):
    super().__init__(name)
    method_name = f'execute_{self.name}'
    method = getattr(self, method_name, self.no_visit_method)
  def no_visit_method(self, node, context):
    raise Exception(f'No execute_{self.name} method defined')

  def copy(self):
    copy = BuiltInFunction(self.name)
    copy.set_context(self.context)
    copy.set_pos(self.starting_position, self.ending_position)
    return copy
  def __repr__(self):
    return f"<built-in function {self.name}>"
  def execute_input_int(self, exec_ctx):
    while True:
      text = input()
      try:
        number = int(text)
        break
      except ValueError:
        print(f"'{text}' must be an integer. Try again!")

  def execute_is_string(self, exec_ctx):
    is_number = isinstance(exec_ctx.symbol_table.get("value"), String)
  execute_is_string.argument_names = ["value"]
  def execute_is_function(self, exec_ctx):
    is_number = isinstance(exec_ctx.symbol_table.get("value"), BaseFunction)
  execute_is_function.argument_names = ["value"]
  def execute_append(self, exec_ctx):
    list_ = exec_ctx.symbol_table.get("list")
    value = exec_ctx.symbol_table.get("value")
  def execute_pop(self, exec_ctx):
    list_ = exec_ctx.symbol_table.get("list")
    index = exec_ctx.symbol_table.get("index")

    fn = fn.value

#This is the result of parser with the function Parse_Outcome

class Parse_Outcome:
  def __init__(self):
    self.error = None
    self.node = None
    self.last_registered_advance_count = 0
    self.advance_count = 0
    self.to_reverse_count = 0

  def success(self, node):
    self.node = node
    return self

  def failure(self, error):
    if not self.error or self.last_registered_advance_count == 0:
      self.error = error
    return self
    
#These are the BuildInFunction that are defined
BuiltInFunction.print       = BuiltInFunction("print")
BuiltInFunction.input       = BuiltInFunction("input")
BuiltInFunction.input_int   = BuiltInFunction("input_int")
BuiltInFunction.clear       = BuiltInFunction("clear")
BuiltInFunction.is_number   = BuiltInFunction("is_number")
BuiltInFunction.is_string   = BuiltInFunction("is_string")
BuiltInFunction.is_list     = BuiltInFunction("is_list")
BuiltInFunction.is_function = BuiltInFunction("is_function")
BuiltInFunction.run			= BuiltInFunction("run")

#Context Class
class Context:
  def __init__(self, display_name, parent=None, parent_entry_pos=None):
    self.display_name = display_name
    self.parent = parent
    self.parent_entry_position = parent_entry_pos
    self.symbol_table = None

#SymbolTable class
class SymbolTable:
  def __init__(self, parent=None):
    self.symbols = {}
    self.parent = parent
  def get(self, name):
    value = self.symbols.get(name, None)
    if value == None and self.parent:
      return self.parent.get(name)
    return value
  def set(self, name, value):
    self.symbols[name] = value
  def remove(self, name):
    del self.symbols[name]

#Interpreter Class 
class Interpreter:
  def visit(self, node, context):
    method_name = f'visit_{type(node).__name__}'
    method = getattr(self, method_name, self.no_visit_method)
    return method(node, context)

  def no_visit_method(self, node, context):
    raise Exception(f'No visit_{type(node).__name__} method defined')

# RunTime Result of Parser
class RunTime_Result:
  def __init__(self):
    self.reset()

def visit_BinOpNode(self, node, context):
    res = RunTime_Result()
    left = res.register(self.visit(node.left_node, context))
    if res.should_return(): 
      return res
    right = res.register(self.visit(node.right_node, context))
    if res.should_return(): 
      return res
    if node.operator_token.type == Compiler_Add:
      result, error = left.added_to(right)
    elif node.operator_token.type == Compiler_Subtract:
      result, error = left.subbed_by(right)
    elif node.operator_token.type == Compiler_Multiply:
      result, error = left.multed_by(right)
    elif node.operator_token.type == Compiler_Divide:
      result, error = left.dived_by(right)
    elif node.operator_token.type == Compiler_Equal:
      result, error = left.get_comparison_eq(right)
    elif node.operator_token.type == Compiler_LessThan:
      result, error = left.get_comparison_lt(right)
    elif node.operator_token.type == Compiler_GreaterThan:
      result, error = left.get_comparison_gt(right)
    elif node.operator_token.matches(Compiler_Keyword, 'AND'):
      result, error = left.anded_by(right)
    elif node.operator_token.matches(Compiler_Keyword, 'OR'):
      result, error = left.ored_by(right)

    if error:
      return res.failure(error)
    else:
      return res.success(result.set_pos(node.pos_start, node.pos_end))

# Phase-4: Intermediate Code Generation Phase

compiler_symbol_table = SymbolTable()
compiler_symbol_table.set("print", BuiltInFunction.print)
compiler_symbol_table.set("input", BuiltInFunction.input)
compiler_symbol_table.set("input_int", BuiltInFunction.input_int)
compiler_symbol_table.set("clear", BuiltInFunction.clear)
compiler_symbol_table.set("is_number", BuiltInFunction.is_number)
compiler_symbol_table.set("is_string", BuiltInFunction.is_string)
compiler_symbol_table.set("is_list", BuiltInFunction.is_list)
compiler_symbol_table.set("is_function", BuiltInFunction.is_function)
compiler_symbol_table.set("run", BuiltInFunction.run)

def run(fn, text):
  # Initiate Tokenss
  lexer = Lexical_Analysis(fn, text)
  Tokenss, error = lexer.make_Tokenss()
  if error: return None, error
  # Create AST
  parser = Parser(Tokenss)
  ast = parser.parse()
  if ast.error: return None, ast.error
  # Run the compiler: Program
  interpreter = Interpreter()
  context = Context('<Program>')
  result = interpreter.visit(ast.node, context)
  return result.value, result.error

# I have created the compiler which includes four phases of compiler that is: 
       
       # Phase-1: Lexical Analysis
       # Phase-2: Syntax Analysis
       # Phase-3: Semantic Analysis
       # Phase-4: Intermediate Code Generation

# This is the Python Shell for my Compiler to execute the compiler and run the code upon the same and this includes the generation of code.

import compiler
while True:
	text = input('compiler >>> ')
	if text.strip() == "": continue
	result, error = compiler.run('<stdin>', text)

	if error:
		print(error.as_string())
	elif result:
		if len(result.elements) == 1:
			print(repr(result.elements[0]))
		else:
			print(repr(result))

# This was the compiler created by me (Name: Ritik Jaiswal | SAP ID: 500084079 | Roll No.: R214220948)
