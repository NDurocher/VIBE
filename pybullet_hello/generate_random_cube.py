"""
This is testing the function of generating .urdf files of random boxes

"""

import os
import random

import numpy as np
# import xml.etree.ElementTree as ET
import lxml.etree as et

def generate_random_urdf_box(save_path, how_many):
    """
    generate 10 boxes - to change the limits of dimensions modify $rand_box_shape$
    source: http://wiki.ros.org/urdf/Tutorials/Adding%20Physical%20and%20Collision%20Properties%20to%20a%20URDF%20Model
    """
    scale = 0.03
    cube_height = 0.03
    gripper_width_limit = 0.055

    for i in range(how_many):
        rand_box_shape = (random.uniform(0.01, gripper_width_limit), random.uniform(0.01, gripper_width_limit), cube_height)

        size_dict = {"size": str(rand_box_shape[0]) + " " + str(rand_box_shape[1]) + " " + str(rand_box_shape[2])}
        print("size_dict", size_dict)
        color_dict = {"rgba": str(random.uniform(0, 1)) + " " + str(random.uniform(0, 1)) + " " + str(random.uniform(0, 1))+ " 1"}

        robot_et = et.Element('robot', {'name': 'random_box'})
        # link name="box"
        link_et = et.SubElement(robot_et, 'link', {'name': 'box'})
        visual_et = et.SubElement(link_et, 'visual')
        geometry_et = et.SubElement(visual_et, 'geometry')
        box_et = et.SubElement(geometry_et, 'box', size_dict )
        material_et = et.SubElement(visual_et, 'material', {"name": "color"})
        color_et = et.SubElement(material_et, 'color', color_dict)

        collision_et = et.SubElement(link_et, 'collision')
        geometry_col_et = et.SubElement(collision_et, 'geometry')
        box_col_et = et.SubElement(geometry_col_et, 'box', size_dict)

        tree = et.ElementTree(robot_et)
        tree.write(save_path + 'box_' + str(i) + '.urdf', pretty_print=True, xml_declaration=True, encoding="utf-8")


if __name__ == "__main__":
    # needs to have the directoery! "generated_urdfs"
    generate_random_urdf_box(save_path='./generated_urdfs/', how_many=10)