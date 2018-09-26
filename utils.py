import colorama

class block(object):
	def __init__(self, header, size=80):
		pre = size - len(header)
		pre //= 2
		post = pre
		print('{} {} {}'.format('=' * (pre - 1), header, '=' * (post - 1)))

	def print(self, string, indent=True):
		print('{}{}'.format(' ' if indent else '', string))

	def __enter__(self):
		return self

	def __exit__(self, type, value, traceback):
	  print()