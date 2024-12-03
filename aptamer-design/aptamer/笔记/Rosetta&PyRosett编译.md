# Rosetta
**1.** https://www.rosettacommons.org/software/academic ,从此处下载rosetta源码进行编译；

**2.** The downloaded file is in form of tar archive with .tgz extension. In a linux or mac, you can untar/uncompress the file by either double clicking on it or run this command in your terminal:
> tar -xvzf rosetta[releasenumber].tar.gz
Unfortunately, currently there is no support for the whole Rosetta on Windows. Dual booting or virtual machines running Linux/MacOS are options.
一般是新建一个文件夹，把源码放到该文件夹下面，再进入终端进行后续操作

**3.** 解压完成之后，进入source文件夹；Open the file that you unzipped and navigate through the folders: Rosetta -> main -> source or use the following bash command:

> cd rosetta*/main/source

之后运行下面的命令进行编译即可，请注意，一定要是使用py3.8进行编译，或者conda/env里面的python解释器为3.8版本的，否则回报错。
Now you can build Rosetta using this general command line (make sure you are in the source folder)

> ./scons.py -j <number_of_cores_to_use> mode=release bin
-j is indicating how many cores you want to use. This number depends on your computer. For example the command below uses 20 cores to build:

> ./scons.py -j 20 mode=release bin
Expect a long time for the compilation to finish, several hours on one core.

**4.**之后等待编译运行完成，这一过程会持续很久很久