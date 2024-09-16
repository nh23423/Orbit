
while True:
    try:
        part = 'gif -index{}'.format(frame_index)
        frame = PhotoImage(file='sa.gif', format=part)
    except:
        print('break')
        self.last_frame = self.frame_index - 1
        break
    self.framelist.append(self.frame)
    self.frame_index += 1
print(self.framelist)