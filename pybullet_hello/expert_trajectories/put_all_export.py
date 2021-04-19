import os

if __name__ == "__main__":

    print(os.getcwd())
    # path_to_save = os.getcwd() + "/expert_trajectories/try_smart_fast"
    dirs = os.listdir(os.getcwd() + '/.')
    for dir in dirs:
        if dir.startswith('try_smart_fast_p'):
            path_now = os.getcwd() + '/' + dir
            print("path_now = ", path_now)
            # go to every dir and rename every picture
            for dir_try in os.listdir(path_now + "/."):
                # print("dir_try = ", dir_try)
                path_dir_try = path_now + "/" + dir_try
                # print("path_dir_try = ", path_dir_try)

                action_path = os.listdir(path_dir_try)
                # print(len(action_path))
                for img in action_path:
                    # print("img", img)
                    img_path = str(path_dir_try) + "/" + str(img)
                    print("img_path", img_path)
                    # .strip('.')[0]
                    new_name = img_path.split('.')[0] + str(dir.split("_")[-1]) + ".jpg"

                    # puts the files into EXPORT directory!
                    new_loc = new_name.replace(dir, 'export')
                    print("new_loc", new_loc)
                    os.rename(img_path, new_loc)
                    # exit(1)


