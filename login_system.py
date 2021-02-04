users = {}
status = ""


def displaymenu():
    status = input("\nRegistered Before?: y/n or press q to quit:\n")
    if status == "y":
        olduser()
    elif status == "n":
        newuser()


def olduser():
    username = input("Username: ")
    passw = input("Password: ")
    for line in open("info.txt", "r").readlines():
        login_info = line.split()
        if username == login_info[0] and passw == login_info[1]:
            print("Username is correct!")
            break
        else:
            print("Wrong username or password!")
            return False


def newuser():
    createlogin = input("Create Username: ")
    if createlogin in users:
        print("Username already exists")
    else:
        createpsw = input("Create Password: ")
        users[createlogin] = createpsw
        print("\n User created\n")
    createuser = open("info.txt", "a")
    createuser.write(createlogin)
    createuser.write(" ")
    createuser.write(createpsw)
    createuser.write("\n")
    createuser.close()


while status != "q":
    displaymenu()