from tkinter import *
import os

#os.system("dem.gif")
root=Tk()
root.geometry('700x570')
frame = Frame(root, relief=RIDGE, borderwidth=2)
frame.pack(fill=BOTH,expand=1)
root.title('Welcome To Thermal Face Recognition System!!!')
frame.config(background='light blue')
label = Label(frame, text="Thermal Face Recognition System",bg='light blue',font=('Times 35 bold'))
label.pack(side=TOP)
filename = PhotoImage(file="demo.png")
background_label = Label(frame,image=filename)
background_label.pack(side=TOP)


def registration():
    exec(open('index.py').read())

def recognition():
    exec(open('recognition.py').read())

def exit():
    root.destroy()


menu = Menu(root)
root.config(menu=menu)

but1=Button(frame,padx=5,pady=5,width=55,bg='white',fg='black',relief=GROOVE,command=registration,text='New Registration',font=('helvetica 15 bold'))
but1.place(x=5,y=100)

but3=Button(frame,padx=5,pady=5,width=55,bg='white',fg='black',relief=GROOVE,command=recognition,text='Recognize',font=('helvetica 15 bold'))
but3.place(x=5,y=200)

but3=Button(frame,padx=5,pady=5,width=55,bg='white',fg='black',relief=GROOVE,command=recognition,text='History',font=('helvetica 15 bold'))
but3.place(x=5,y=300)

but5=Button(frame,padx=5,pady=5,width=5,bg='white',fg='black',relief=GROOVE,text='EXIT',command=exit,font=('helvetica 15 bold'))
but5.place(x=320,y=440)

root.mainloop()