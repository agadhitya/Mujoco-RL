#!/usr/bin/env python3

#Copyright 2015 Yale University - Grablab
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import sys
import json
import urllib
import urllib.request as urllib2
import rospkg

baseDir = rospkg.RosPack().get_path('active_vision')
output_directory = baseDir+"/models/ycbAV"

# You can either set this to "all" or a list of the objects that you'd like to
# download.
# objects_to_download = "all"
objects_to_download = ["003_cracker_box",
                       "005_tomato_soup_can",
                       "006_mustard_bottle",
                       "008_pudding_box",
                       "010_potted_meat_can",
                       "013_apple",
                       "021_bleach_cleanser",
                       "024_bowl",
                       "025_mug",
                       "035_power_drill",
                       "055_baseball",
                       "072-a_toy_airplane"]

# You can edit this list to only download certain kinds of files.
# 'berkeley_rgbd' contains all of the depth maps and images from the Carmines.
# 'berkeley_rgb_highres' contains all of the high-res images from the Canon cameras.
# 'berkeley_processed' contains all of the segmented point clouds and textured meshes.
# 'google_16k' contains google meshes with 16k vertices.
# 'google_64k' contains google meshes with 64k vertices.
# 'google_512k' contains google meshes with 512k vertices.
# See the website for more details.
#files_to_download = ["berkeley_rgbd", "berkeley_rgb_highres", "berkeley_processed", "google_16k", "google_64k", "google_512k"]
files_to_download = ["google_16k"]
file_type = files_to_download[0]

# Extract all files from the downloaded .tgz, and remove .tgz files.
# If false, will just download all .tgz files to output_directory
extract = True

base_url = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/"
objects_url = base_url + "objects.json"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def fetch_objects(url):
    response = urllib2.urlopen(url)
    html = response.read()
    objects = json.loads(html)
    return objects["objects"]

def download_file(url, filename):
    u = urllib2.urlopen(url)
    f = open(filename, 'wb')
    meta = u.info()
    file_size = int(meta.get("Content-Length")[0])
    print("Downloading: %s (%s MB)" % (filename, file_size/1000000.0))

    file_size_dl = 0
    block_sz = 65536
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl/1000000.0, file_size_dl * 100. / file_size)
        status = status + chr(8)*(len(status)+1)
        print(status),
    f.close()

def tgz_url(object, type):
    if type in ["berkeley_rgbd", "berkeley_rgb_highres"]:
        return base_url + "berkeley/{object}/{object}_{type}.tgz".format(object=object,type=type)
    elif type in ["berkeley_processed"]:
        return base_url + "berkeley/{object}/{object}_berkeley_meshes.tgz".format(object=object,type=type)
    else:
        return base_url + "google/{object}_{type}.tgz".format(object=object,type=type)

def extract_tgz(filename, dir):
    tar_command = "tar -xzf {filename} -C {dir}".format(filename=filename,dir=dir)
    os.system(tar_command)
    os.remove(filename)

def check_url(url):
    try:
        request = urllib2.Request(url)
        request.get_method = lambda : 'HEAD'
        response = urllib2.urlopen(request)
        return True
    except Exception as e:
        return False

def generate_sdf(object_name, data_type, object_mass, x, y, z, ixx, ixy, ixz, iyy, iyz, izz ,data_path):
    file_name = "YCB"+object_name.split("_")[0]
    if os.access(data_path+"/sdf", os.F_OK) == False:
    	try:
    		os.mkdir(data_path+ "/sdf")
    	except OSError as mkdir_error:
    		print("[ycb_benchmarks] Failed to create folder %s. Error message: %s" % (data_path+"/sdf", str(mkdir_error)))
    		exit(1)

    f = open(data_path + "/sdf/" + file_name + ".sdf", 'w')

    sdf_str = """<?xml version="1.0" ?>
<sdf version="1.5">
  <model name=\"""" + file_name + """\">
    <link name="link">
      <inertial>
        <pose>0 0 0 0 0 0</pose>
        <inertia>
          <ixx>""" + str(ixx) + """</ixx>
          <ixy>""" + str(ixy) + """</ixy>
          <ixz>""" + str(ixz) + """</ixz>
          <iyy>""" + str(iyy) + """</iyy>
          <iyz>""" + str(iyz) + """</iyz>
          <izz>""" + str(izz) + """</izz>
        </inertia>
        <mass>""" + str(object_mass) + """</mass>
      </inertial>
      <collision name="collision">
        <pose>""" + str(-x) + """ """ + str(-y) + """ """ + str(-z) + """ 0 0 0</pose>
        <geometry>
          <mesh>
            <uri>file:///""" + data_path + """/""" + object_name + """/""" + data_type + """/nontextured.stl</uri>
          </mesh>
        </geometry>
        <surface>
          <contact>
            <ode>
              <max_vel>0.1</max_vel>
              <min_depth>0.001</min_depth>
              <soft_erp>0.5</soft_erp>
              <soft_cfm>0.001</soft_cfm>
            </ode>
          </contact>
          <friction>
            <ode>
              <mu>10000.0</mu>
              <mu2>10000.0</mu2>
            </ode>
          </friction>
        </surface>
      </collision>
      <visual name="visual">
        <pose>""" + str(-x) + """ """ + str(-y) + """ """ + str(-z) + """ 0 0 0</pose>
        <geometry>
          <mesh>
            <uri>file:///""" + data_path + """/""" + object_name + """/""" + data_type + """/nontextured.stl</uri>
          </mesh>
        </geometry>
    	<material>
          <ambient>1 0 0 1</ambient>
          <diffuse>1 0 0 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
          <emissive>0 0 0 0</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""
    f.write(sdf_str)
    f.close()

if __name__ == "__main__":

    for object in objects_to_download:
        url = tgz_url(object, file_type)
        if not check_url(url):
            continue
        filename = "{path}/{object}_{file_type}.tgz".format(path=output_directory,
                                                            object=object,
                                                            file_type=file_type)

        if os.access(output_directory+"/"+object, os.F_OK) == False:
            download_file(url, filename)
            if extract:
                extract_tgz(filename, output_directory)

    file_object_ids = open(output_directory + "/object_ids.txt", 'r')
    for object in objects_to_download:
        for file_line in file_object_ids:
            if object in file_line:
                object_name = file_line.split()[0]
                object_mass = file_line.split()[1]
                if len(file_line.split()) == 11:
                    x = float(file_line.split()[2])
                    y = float(file_line.split()[3])
                    z = float(file_line.split()[4])
                    ixx = file_line.split()[5]
                    ixy = file_line.split()[6]
                    ixz = file_line.split()[7]
                    iyy = file_line.split()[8]
                    iyz = file_line.split()[9]
                    izz = file_line.split()[10]
                else:
                    x = 0
                    y = 0
                    z = 0
                    ixx = 1.0
                    ixy = 0.0
                    ixz = 0.0
                    iyy = 1.0
                    iyz = 0.0
                    izz = 1.0
                break
        generate_sdf(object_name,file_type,object_mass,x,y,z,ixx,ixy,ixz,iyy,iyz,izz,output_directory)
