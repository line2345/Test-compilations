import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class MyTurtleControl(Node):
    def __init__(self):
        super().__init__('my_turtle_control')
        self.publisher_ = self.create_publisher(Twist, 'turtle1/cmd_vel', 10)
        self.timer_ = self.create_timer(1.0, self.publish_velocity)

    def publish_velocity(self):
        twist = Twist()
        # 设置小乌龟的线速度和角速度，以画出五角星
        twist.linear.x = 1.0
        twist.angular.z = 0.8
        self.publisher_.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    my_turtle_control = MyTurtleControl()
    rclpy.spin(my_turtle_control)
    rclpy.shutdown()

if __name__ == '__main__':
    main()


mv /mnt/c/Users/86198/Downloads/ncbi-blast-2.13.0+-x64-arm-linux.tar.gz /home/bio_kang/software/
