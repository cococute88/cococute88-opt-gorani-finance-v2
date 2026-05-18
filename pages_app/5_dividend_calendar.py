# Dividend calendar wrapper page

from importlib import import_module

render_calendar = import_module("modules.dividend_calendar").render
render_calendar()
