import sys

file_path = r"c:\STAIR_Digital\pdf_agent\ui\chat_panel.py"

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

start_idx = -1
for i, line in enumerate(lines):
    if 'with st.chat_message("assistant"):' in line:
        start_idx = i + 1
        break

if start_idx != -1:
    for i in range(start_idx, len(lines)):
        if lines[i].startswith('    finally:'):
            break
        if lines[i].strip() or lines[i].startswith(' '): # If not entirely empty
            if not lines[i].startswith(' ' * 12) and lines[i].startswith(' ' * 8) and not lines[i].strip() == '':
                 lines[i] = '    ' + lines[i]
            elif lines[i].startswith(' ' * 8):
                 lines[i] = '    ' + lines[i]
            elif lines[i].startswith(' ' * 12):
                 lines[i] = '    ' + lines[i]
            elif lines[i].startswith(' ' * 16):
                 lines[i] = '    ' + lines[i]
            elif lines[i].startswith(' ' * 20):
                 lines[i] = '    ' + lines[i]
            elif lines[i].startswith(' ' * 24):
                 lines[i] = '    ' + lines[i]
            elif lines[i].startswith(' ' * 28):
                 lines[i] = '    ' + lines[i]
            elif lines[i].startswith(' ' * 32):
                 lines[i] = '    ' + lines[i]
            # Just blindly add 4 spaces to everything in the block except totally blank lines
            elif lines[i] != '\n':
                 lines[i] = '    ' + lines[i]

with open(file_path, "w", encoding="utf-8") as f:
    f.writelines(lines)
