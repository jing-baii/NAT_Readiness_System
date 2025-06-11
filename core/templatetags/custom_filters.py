# -*- coding: utf-8 -*-
from django import template
import re

register = template.Library()

@register.filter
def youtube_id(url):
    """Extract YouTube video ID from various YouTube URL formats."""
    if not url:
        return None
        
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?]+)',  # Standard YouTube URLs
        r'youtube\.com\/embed\/([^&\n?]+)',  # Embed URLs
        r'youtube\.com\/v\/([^&\n?]+)',  # Old format URLs
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

@register.filter
def divide(value, arg):
    """Divide the value by the argument"""
    try:
        return float(value) / float(arg)
    except (ValueError, ZeroDivisionError):
        return 0

@register.filter
def split(value, delimiter=','):
    """Split a string by the given delimiter"""
    return value.split(delimiter)

@register.filter
def mul(value, arg):
    """Multiply the value by the argument"""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return 0

@register.filter
def filter_subject(queryset, subject_id):
    return queryset.filter(subtopic__general_topic__subject__id=subject_id)

@register.filter
def get_item(dictionary, key):
    """Get an item from a dictionary safely."""
    if isinstance(dictionary, dict):
        return dictionary.get(str(key), 0)  # Return 0 if key doesn't exist
    return 0  # Return 0 if not a dict

# Add these for Likert analytics
@register.filter
def get_total(dictionary):
    """
    Return the sum of all values in the dictionary.
    Usage: {{ dictionary|get_total }}
    """
    if not dictionary:
        return 0
    return sum(float(v) for v in dictionary.values())

@register.filter
def get_weighted_total(dictionary):
    """
    Return the sum of key*value for all items in the dictionary (keys are Likert weights).
    Usage: {{ dictionary|get_weighted_total }}
    """
    if not dictionary:
        return 0
    return sum(float(k) * float(v) for k, v in dictionary.items())

@register.filter
def multiply(value, arg):
    """Multiplies the value by the argument."""
    try:
        return int(value) * int(arg)
    except (ValueError, TypeError):
        return 0 