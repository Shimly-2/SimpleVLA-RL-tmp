from codetiming import Timer

import sys
from typing import Optional

class ColorPrinter:
    """终端彩色输出工具类"""
    
    # ANSI颜色代码
    COLORS = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
    }
    
    # 样式代码
    STYLES = {
        'bold': '\033[1m',
        'dim': '\033[2m',
        'italic': '\033[3m',
        'underline': '\033[4m',
        'blink': '\033[5m',
        'reverse': '\033[7m',
        'strikethrough': '\033[9m',
    }
    
    # 重置代码
    RESET = '\033[0m'
    
    @classmethod
    def supports_color(cls) -> bool:
        """检查终端是否支持颜色"""
        return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    
    @classmethod
    def colorize(cls, text: str, color: str = None, style: str = None) -> str:
        """给文本添加颜色和样式"""
        if not cls.supports_color():
            return text
            
        codes = []
        
        if color and color in cls.COLORS:
            codes.append(cls.COLORS[color])
        
        if style and style in cls.STYLES:
            codes.append(cls.STYLES[style])
        
        if codes:
            return ''.join(codes) + text + cls.RESET
        return text
    
    @classmethod
    def print_colored(cls, text: str, color: str = None, style: str = None, **kwargs):
        """打印彩色文本"""
        colored_text = cls.colorize(text, color, style)
        print(colored_text, **kwargs)
        
class ColorTimer:
    """支持彩色输出的计时器"""
    
    def __init__(self, name: str = "Timer", text: str = "{name}: {seconds:.3f} seconds", 
                 color: str = None, style: str = None):
        self.name = name
        self.text = text
        self.color = color
        self.style = style
        self.start_time = None
        self.elapsed_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        import time
        self.elapsed_time = time.perf_counter() - self.start_time
        message = self.text.format(name=self.name, seconds=self.elapsed_time)
        ColorPrinter.print_colored(message, self.color, self.style)

with Timer(name='gen', text="{name}: {seconds:.1f} seconds", color="red") as timer:
    for i in range(30):
        target = int(256 * (1 + 1 / 20 * i))
        target = min(target, 511)
        
        print(target)