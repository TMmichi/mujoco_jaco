# from pynput import keyboard

# print('Press s or n to continue:')

# with keyboard.Events() as events:
#     # Block for as much as possible
#     event = events.get(1e6)
#     print(event.key)
#     if event.key == keyboard.KeyCode.from_char('s'):
#         print("YES")
#     else:
#         print("NO")

import curses, time

def input_char(message):
    try:
        win = curses.initscr()
        win.addstr(0, 0, message)
        while True: 
            ch = win.getch()
            if ch in range(32, 127): 
                break
            time.sleep(0.05)
    finally:
        curses.endwin()
    return chr(ch)

c = input_char('Do you want to continue? y/[n]')
if c.lower() in ['y', 'yes']:
    print('yes')
else:
    print('no (got {})'.format(c))