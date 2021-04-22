import os

if __name__ == "__main__":

    path_now = os.getcwd()
    print("os.getcwd() = ", os.getcwd())
    path_here = "/".join(path_now.split("/")[:-1])
    print("path_here = ", path_here)

    action_path = os.listdir(path_now)
    for id, file in enumerate(action_path):
        if file.endswith(".py"):
            action_path.pop(id)
    print("action_path = ", action_path)
    for grasp_release in action_path:
        print("grasp_release = ", grasp_release)
        path_g_r = path_now + "/" + grasp_release
        print("path_g_r= ", path_g_r)
        for i, img_name_before in enumerate(os.listdir(path_g_r)):
            print("img_name_before = ", img_name_before)
            img_path = str(path_g_r) + "/" + str(img_name_before)
            print("img_path", img_path)
            # .strip('.')[0]
            new_name = img_path.split('.')[0] + "_" + str(i) + ".jpg"
            new_name = new_name.replace("grasp_release", 'export')
            print("new_name = ", new_name)
            # puts the files into EXPORT directory!
            os.rename(img_path, new_name)
            # exit(1)
