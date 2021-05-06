import os

if __name__ == "__main__":

    print(os.getcwd())
    # path_to_save = os.getcwd() + "/expert_trajectories/try_smart_fast"
    dirs = os.listdir(os.getcwd() + '/.')
    for dir in dirs:
        # try_smart_fast_p_50, try_smart_fast_p_100, try_smart_fast_p_150
        # wanted_dirs = ('try_smart_fast_p50', 'try_smart_fast_p100', 'try_smart_fast_p150')
        # wanted_dirs = ('try_smart_fast_p200')
        wanted_dirs = ('try_smart_fast_p250')

        if dir in wanted_dirs:
        # if dir == 'try_smart_fast_p50test':
            path_now = os.getcwd() + '/' + dir
            print("path_now = ", path_now)

            dir_new = dir + '_without_similar'
            path_dir_without_sim = path_now + '_without_similar'

            if not os.path.exists(path_dir_without_sim):
                os.mkdir(path_dir_without_sim)
                os.mkdir(path_dir_without_sim + '/north')
                os.mkdir(path_dir_without_sim + '/south')
                os.mkdir(path_dir_without_sim + '/east')
                os.mkdir(path_dir_without_sim + '/west')
                os.mkdir(path_dir_without_sim + '/grasp')
                os.mkdir(path_dir_without_sim + '/release')

            print("path_dir_without_sim = ", path_dir_without_sim)

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
                    if not 'similar' in img_path.split('.')[0]:
                        # move to the new directory
                        new_name = img_path.split('.')[0] + "_copied.jpg"
                        import os.path

                        if os.path.isfile(img_path):
                            # puts the files into EXPORT directory!
                            new_loc = new_name.replace(dir, dir_new)
                            print("new_loc", new_loc)
                            command = 'cp ' + img_path + ' ' + new_loc
                            os.system(command)

                            # exit(1)


