import time

from django import template

register = template.Library()

@register.simple_tag
def multiply(arg):
    # you would need to do any localization of the result here
    return arg * 2