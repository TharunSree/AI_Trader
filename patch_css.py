import re

with open('templates/partials/head-css.html', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace the booting state CSS
old_css = '''  .jarvis-booting .glass, 
  .jarvis-booting .content-page .col-xl-3, 
  .jarvis-booting .content-page .col-xl-4,
  .jarvis-booting .content-page .col-xl-6,
  .jarvis-booting .content-page .col-xl-8,
  .jarvis-booting .content-page .col-12,
  .jarvis-booting .content-page .col-md-6 {
      opacity: 0 !important;
      transform: translateY(30px) !important;
  }'''

new_css = '''  .jarvis-booting .glass, 
  .jarvis-booting .content-page .col-xl-3, 
  .jarvis-booting .content-page .col-xl-4,
  .jarvis-booting .content-page .col-xl-6,
  .jarvis-booting .content-page .col-xl-8,
  .jarvis-booting .content-page .col-12,
  .jarvis-booting .content-page .col-md-6,
  .jarvis-booting .card {
      opacity: 0 !important;
      transform: translateY(40px) scale(0.97) !important;
      clip-path: inset(100% 0 0 0) !important;
      filter: brightness(2) contrast(1.5) sepia(1) hue-rotate(180deg) saturate(3) !important;
  }'''
text = text.replace(old_css, new_css)

with open('templates/partials/head-css.html', 'w', encoding='utf-8') as f:
    f.write(text)
