覆盖折线的功能接口
先用run.sh编译生成可执行文件polyline
再运行polyline
具体使用时，先把需要用不同级别方块覆盖的区域放在不同的文件夹下，比如1公里放一个文件夹，5公里放另一个文件夹，8公里的再放一个文件夹...
然后调用类似/home/gml/s2geometry/kye/run.sh 的脚本来执行，通过传递不同的参数，把输出重定向到各个文件，最后输出是一个一个json文件，
当然根据需求，也可以将输出格式改成其他形式，比如csv
