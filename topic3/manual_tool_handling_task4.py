"""
Manual Tool Calling Exercise
Students will see how tool calling works under the hood.

This version includes:
- Weather tool (simulated)
- Calculator tool with geometric functions (custom implementation)
"""

import json
import math
from openai import OpenAI

# ============================================
# PART 1: Define Your Tools
# ============================================

def get_weather(location: str) -> str:
    """Get the current weather for a location"""
    # Simulated weather data
    weather_data = {
        "San Francisco": "Sunny, 72°F",
        "New York": "Cloudy, 55°F",
        "London": "Rainy, 48°F",
        "Tokyo": "Clear, 65°F"
    }
    return weather_data.get(location, f"Weather data not available for {location}")


def calculator(operation: str, params: str) -> str:
    """
    A calculator tool with arithmetic and geometric functions.
    
    Args:
        operation: The type of calculation to perform. Options:
            Arithmetic: "add", "subtract", "multiply", "divide", "power", "sqrt", "evaluate"
            Geometry 2D: "circle_area", "circle_circumference", "rectangle_area", 
                        "rectangle_perimeter", "triangle_area", "triangle_perimeter"
            Geometry 3D: "sphere_volume", "sphere_surface", "cylinder_volume", 
                        "cylinder_surface", "cone_volume", "box_volume", "box_surface"
            Trigonometry: "sin", "cos", "tan", "degrees_to_radians", "radians_to_degrees"
        params: JSON string with the required parameters for the operation.
    
    Returns:
        JSON string with the result or error message.
    """
    try:
        # Parse the input parameters using json.loads
        args = json.loads(params)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON parameters: {e}"})
    
    result = None
    
    try:
        # ==========================================
        # ARITHMETIC OPERATIONS
        # ==========================================
        if operation == "add":
            # params: {"a": number, "b": number}
            result = args["a"] + args["b"]
            
        elif operation == "subtract":
            # params: {"a": number, "b": number}
            result = args["a"] - args["b"]
            
        elif operation == "multiply":
            # params: {"a": number, "b": number}
            result = args["a"] * args["b"]
            
        elif operation == "divide":
            # params: {"a": number, "b": number}
            if args["b"] == 0:
                return json.dumps({"error": "Division by zero"})
            result = args["a"] / args["b"]
            
        elif operation == "power":
            # params: {"base": number, "exponent": number}
            result = math.pow(args["base"], args["exponent"])
            
        elif operation == "sqrt":
            # params: {"value": number}
            if args["value"] < 0:
                return json.dumps({"error": "Cannot take square root of negative number"})
            result = math.sqrt(args["value"])
            
        elif operation == "evaluate":
            # params: {"expression": string}
            # Safely evaluate a mathematical expression using ast.literal_eval
            # For more complex expressions, we use a restricted eval approach
            import ast
            import operator
            
            # Define allowed operations for safe evaluation
            allowed_operators = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Pow: operator.pow,
                ast.USub: operator.neg,
                ast.UAdd: operator.pos,
            }
            
            def safe_eval(node):
                if isinstance(node, ast.Constant):  # Numbers
                    return node.value
                elif isinstance(node, ast.BinOp):  # Binary operations
                    left = safe_eval(node.left)
                    right = safe_eval(node.right)
                    op_type = type(node.op)
                    if op_type in allowed_operators:
                        return allowed_operators[op_type](left, right)
                    else:
                        raise ValueError(f"Unsupported operator: {op_type}")
                elif isinstance(node, ast.UnaryOp):  # Unary operations
                    operand = safe_eval(node.operand)
                    op_type = type(node.op)
                    if op_type in allowed_operators:
                        return allowed_operators[op_type](operand)
                    else:
                        raise ValueError(f"Unsupported operator: {op_type}")
                elif isinstance(node, ast.Expression):
                    return safe_eval(node.body)
                else:
                    raise ValueError(f"Unsupported expression type: {type(node)}")
            
            expression = args["expression"]
            tree = ast.parse(expression, mode='eval')
            result = safe_eval(tree)
        
        # ==========================================
        # 2D GEOMETRY
        # ==========================================
        elif operation == "circle_area":
            # params: {"radius": number}
            result = math.pi * args["radius"] ** 2
            
        elif operation == "circle_circumference":
            # params: {"radius": number}
            result = 2 * math.pi * args["radius"]
            
        elif operation == "rectangle_area":
            # params: {"length": number, "width": number}
            result = args["length"] * args["width"]
            
        elif operation == "rectangle_perimeter":
            # params: {"length": number, "width": number}
            result = 2 * (args["length"] + args["width"])
            
        elif operation == "triangle_area":
            # params: {"base": number, "height": number}
            # OR params: {"a": number, "b": number, "c": number} for Heron's formula
            if "base" in args and "height" in args:
                result = 0.5 * args["base"] * args["height"]
            elif "a" in args and "b" in args and "c" in args:
                # Heron's formula
                a, b, c = args["a"], args["b"], args["c"]
                s = (a + b + c) / 2  # Semi-perimeter
                if s <= a or s <= b or s <= c:
                    return json.dumps({"error": "Invalid triangle: sides don't form a valid triangle"})
                result = math.sqrt(s * (s - a) * (s - b) * (s - c))
            else:
                return json.dumps({"error": "triangle_area requires either {base, height} or {a, b, c}"})
                
        elif operation == "triangle_perimeter":
            # params: {"a": number, "b": number, "c": number}
            result = args["a"] + args["b"] + args["c"]
        
        # ==========================================
        # 3D GEOMETRY
        # ==========================================
        elif operation == "sphere_volume":
            # params: {"radius": number}
            result = (4/3) * math.pi * args["radius"] ** 3
            
        elif operation == "sphere_surface":
            # params: {"radius": number}
            result = 4 * math.pi * args["radius"] ** 2
            
        elif operation == "cylinder_volume":
            # params: {"radius": number, "height": number}
            result = math.pi * args["radius"] ** 2 * args["height"]
            
        elif operation == "cylinder_surface":
            # params: {"radius": number, "height": number}
            # Surface = 2*pi*r*h (lateral) + 2*pi*r^2 (top and bottom)
            r, h = args["radius"], args["height"]
            result = 2 * math.pi * r * h + 2 * math.pi * r ** 2
            
        elif operation == "cone_volume":
            # params: {"radius": number, "height": number}
            result = (1/3) * math.pi * args["radius"] ** 2 * args["height"]
            
        elif operation == "box_volume":
            # params: {"length": number, "width": number, "height": number}
            result = args["length"] * args["width"] * args["height"]
            
        elif operation == "box_surface":
            # params: {"length": number, "width": number, "height": number}
            l, w, h = args["length"], args["width"], args["height"]
            result = 2 * (l*w + w*h + h*l)
        
        # ==========================================
        # TRIGONOMETRY
        # ==========================================
        elif operation == "sin":
            # params: {"angle": number, "unit": "radians" or "degrees"}
            angle = args["angle"]
            if args.get("unit", "radians") == "degrees":
                angle = math.radians(angle)
            result = math.sin(angle)
            
        elif operation == "cos":
            # params: {"angle": number, "unit": "radians" or "degrees"}
            angle = args["angle"]
            if args.get("unit", "radians") == "degrees":
                angle = math.radians(angle)
            result = math.cos(angle)
            
        elif operation == "tan":
            # params: {"angle": number, "unit": "radians" or "degrees"}
            angle = args["angle"]
            if args.get("unit", "radians") == "degrees":
                angle = math.radians(angle)
            result = math.tan(angle)
            
        elif operation == "degrees_to_radians":
            # params: {"degrees": number}
            result = math.radians(args["degrees"])
            
        elif operation == "radians_to_degrees":
            # params: {"radians": number}
            result = math.degrees(args["radians"])
            
        else:
            return json.dumps({"error": f"Unknown operation: {operation}"})
        
        # Format the result using json.dumps
        return json.dumps({
            "operation": operation,
            "params": args,
            "result": round(result, 10) if isinstance(result, float) else result
        })
        
    except KeyError as e:
        return json.dumps({"error": f"Missing required parameter: {e}"})
    except Exception as e:
        return json.dumps({"error": f"Calculation error: {e}"})


def count_letter_occurrences(text: str, letter: str, case_sensitive: bool = False) -> str:
    """Count how many times a letter appears in text."""
    if not letter or len(letter) != 1:
        return json.dumps({"error": "Parameter 'letter' must be exactly one character"})

    search_text = text if case_sensitive else text.lower()
    search_letter = letter if case_sensitive else letter.lower()
    count = search_text.count(search_letter)

    return json.dumps(
        {
            "text": text,
            "letter": letter,
            "case_sensitive": case_sensitive,
            "count": count,
        }
    )


def text_insights(text: str) -> str:
    """A custom tool returning basic text statistics."""
    words = [w for w in text.split() if w.strip()]
    longest_word = max(words, key=len) if words else ""

    return json.dumps(
        {
            "text": text,
            "character_count": len(text),
            "word_count": len(words),
            "longest_word": longest_word,
        }
    )


# ============================================
# PART 2: Describe Tools to the LLM
# ============================================

# This is the JSON schema that tells the LLM what tools exist
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name, e.g. San Francisco"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": """A calculator with arithmetic and geometric functions.
            
Operations available:
- Arithmetic: add, subtract, multiply, divide, power, sqrt, evaluate (for expressions like "2+3*4")
- 2D Geometry: circle_area, circle_circumference, rectangle_area, rectangle_perimeter, triangle_area, triangle_perimeter
- 3D Geometry: sphere_volume, sphere_surface, cylinder_volume, cylinder_surface, cone_volume, box_volume, box_surface
- Trigonometry: sin, cos, tan (with unit: "degrees" or "radians"), degrees_to_radians, radians_to_degrees

The params should be a JSON string with the required parameters for each operation.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The calculation operation to perform",
                        "enum": [
                            "add", "subtract", "multiply", "divide", "power", "sqrt", "evaluate",
                            "circle_area", "circle_circumference", "rectangle_area", "rectangle_perimeter",
                            "triangle_area", "triangle_perimeter",
                            "sphere_volume", "sphere_surface", "cylinder_volume", "cylinder_surface",
                            "cone_volume", "box_volume", "box_surface",
                            "sin", "cos", "tan", "degrees_to_radians", "radians_to_degrees"
                        ]
                    },
                    "params": {
                        "type": "string",
                        "description": """JSON string with operation parameters. Examples:
- add/subtract/multiply/divide: {"a": 10, "b": 5}
- power: {"base": 2, "exponent": 8}
- sqrt: {"value": 16}
- evaluate: {"expression": "2 + 3 * 4"}
- circle_area/circumference: {"radius": 5}
- rectangle_area/perimeter: {"length": 10, "width": 5}
- triangle_area: {"base": 6, "height": 4} or {"a": 3, "b": 4, "c": 5}
- sphere_volume/surface: {"radius": 3}
- cylinder_volume/surface: {"radius": 2, "height": 10}
- cone_volume: {"radius": 3, "height": 6}
- box_volume/surface: {"length": 4, "width": 3, "height": 2}
- sin/cos/tan: {"angle": 45, "unit": "degrees"}
- degrees_to_radians: {"degrees": 180}
- radians_to_degrees: {"radians": 3.14159}"""
                    }
                },
                "required": ["operation", "params"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "count_letter_occurrences",
            "description": "Count occurrences of a single letter in text, with optional case sensitivity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to search within."
                    },
                    "letter": {
                        "type": "string",
                        "description": "A single letter/character to count, e.g. 's'."
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "If true, uppercase/lowercase are treated as different letters.",
                        "default": False
                    }
                },
                "required": ["text", "letter"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "text_insights",
            "description": "Return basic text stats: character count, word count, and longest word.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to analyze."
                    }
                },
                "required": ["text"]
            }
        }
    }
]


# ============================================
# PART 3: The Agent Loop
# ============================================

TOOL_HANDLERS = {
    "get_weather": lambda args: get_weather(**args),
    "calculator": lambda args: calculator(args["operation"], args["params"]),
    "count_letter_occurrences": lambda args: count_letter_occurrences(**args),
    "text_insights": lambda args: text_insights(**args),
}


def execute_tool_call(function_name: str, function_args: dict) -> str:
    """Dispatch tool execution from a centralized registry."""
    tool_handler = TOOL_HANDLERS.get(function_name)
    if tool_handler is None:
        return f"Error: Unknown function {function_name}"

    try:
        return tool_handler(function_args)
    except Exception as e:
        return f"Error executing {function_name}: {e}"


def run_agent(user_query: str):
    """
    Simple agent that can use tools.
    Shows the manual loop that LangGraph automates.
    """
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Start conversation with user query
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided tools when needed."},
        {"role": "user", "content": user_query}
    ]
    
    print(f"User: {user_query}\n")
    
    # Agent loop - can iterate up to 5 times
    for iteration in range(5):
        print(f"--- Iteration {iteration + 1} ---")
        
        # Call the LLM
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,  # â† This tells the LLM what tools are available
            tool_choice="auto"  # Let the model decide whether to use tools
        )
        
        assistant_message = response.choices[0].message
        
        # Check if the LLM wants to call a tool
        if assistant_message.tool_calls:
            print(f"LLM wants to call {len(assistant_message.tool_calls)} tool(s)")
            
            # Add the assistant's response to messages
            messages.append(assistant_message)
            
            # Execute each tool call
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"  Tool: {function_name}")
                print(f"  Args: {function_args}")
                
                result = execute_tool_call(function_name, function_args)
                
                print(f"  Result: {result}")
                
                # Add the tool result back to the conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": result
                })
            
            print()
            # Loop continues - LLM will see the tool results
            
        else:
            # No tool calls - LLM provided a final answer
            print(f"Assistant: {assistant_message.content}\n")
            return assistant_message.content
    
    return "Max iterations reached"


# ============================================
# PART 4: Test It
# ============================================

if __name__ == "__main__":
    # Test query that requires tool use
    # print("="*60)
    # print("TEST 1: Weather Tool")
    # print("="*60)
    # run_agent("What's the weather like in San Francisco?")
    
    # print("\n" + "="*60)
    # print("TEST 2: Query not requiring tool")
    # print("="*60)
    # run_agent("Say hello!")
    
    # print("\n" + "="*60)
    # print("TEST 3: Multiple weather queries")
    # print("="*60)
    # run_agent("What's the weather in New York and London?")
    
    # print("\n" + "="*60)
    # print("TEST 4: Calculator - Basic Arithmetic")
    # print("="*60)
    # run_agent("What is 15 multiplied by 7, then add 25 to the result?")
    
    # print("\n" + "="*60)
    # print("TEST 5: Calculator - Circle Geometry")
    # print("="*60)
    # run_agent("I have a circular pizza with a radius of 8 inches. What is its area and circumference?")
    
    # print("\n" + "="*60)
    # print("TEST 6: Calculator - 3D Geometry (Sphere)")
    # print("="*60)
    # run_agent("Calculate the volume and surface area of a sphere with radius 5 cm.")
    
    # print("\n" + "="*60)
    # print("TEST 7: Calculator - Cylinder")  
    # print("="*60)
    # run_agent("A cylindrical water tank has a radius of 2 meters and height of 10 meters. What is its volume?")
    
    # print("\n" + "="*60)
    # print("TEST 8: Calculator - Triangle (Heron's Formula)")
    # print("="*60)
    # run_agent("Find the area of a triangle with sides 3, 4, and 5 units.")
    
    # print("\n" + "="*60)
    # print("TEST 9: Calculator - Trigonometry")
    # print("="*60)
    # run_agent("What is the sine and cosine of 45 degrees?")
    
    # print("\n" + "="*60)
    # print("TEST 10: Combined Weather + Calculator")
    # print("="*60)
    # run_agent("What's the weather in Tokyo? Also, if I have a rectangular garden that is 12 meters long and 8 meters wide, what is its area and perimeter?")

    # print("\n" + "="*60)
    # print("TEST 11: Letter Counting Tool")
    # print("="*60)
    # run_agent("How many s are in Mississippi riverboats?")

    # print("\n" + "="*60)
    # print("TEST 12: Letter Counting with Case Sensitivity")
    # print("="*60)
    # run_agent("Count how many A letters are in 'Abracadabra' and make it case sensitive.")

    print("\n" + "="*60)
    print("TEST 13: Multiple Tools")
    print("="*60)
    run_agent("What is the sin of the difference between the number of i's and the number of s's in Mississippi riverboats")

    print("\n" + "="*60)
    print("TEST 13: Multiple Tools")
    print("="*60)
    run_agent("Use text_insights on 'Large language models transform productivity', then calculate the square root of the character count using calculator, and report weather in New York.")

