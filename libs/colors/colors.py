import colorama
from colorama import *

colorama.init()

Fore.BRIGHT_GREEN = Style.BRIGHT + Fore.GREEN

class cm(object):
	DIM = Style.DIM

	def __init__(self, *args):
		self.color = [a for a in (args or []) if a is not None]
		self.RESET = ''.join([Style.RESET_ALL] + self.color)

	def __enter__(self):
		print(*self.color, end='')
		return self

	def __exit__(self, type, value, traceback):
	  print(Style.RESET_ALL, end='')