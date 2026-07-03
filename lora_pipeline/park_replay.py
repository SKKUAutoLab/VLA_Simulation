"""녹화 주차 시연 재생: 출발pose 세팅 → 명령시퀀스 재생 → 최종 슬롯 확인."""
import csv,math,rclpy,time
from rclpy.node import Node
from interfaces_pkg.msg import MotionCommand
from gazebo_msgs.srv import GetEntityState, SetEntityState
from rclpy.qos import QoSProfile,QoSReliabilityPolicy,QoSHistoryPolicy,QoSDurabilityPolicy
rows=list(csv.DictReader(open('/tmp/park_demo.csv')))
def nz(r): return int(r['steer'])!=0 or int(r['l'])!=0 or int(r['r'])!=0
drv=[i for i,r in enumerate(rows) if nz(r)]
seq=rows[drv[0]:drv[-1]+1]
s=seq[0]; sx,sy,syaw=float(s['x']),float(s['y']),float(s['yaw'])
rclpy.init();n=Node('rep')
qos=QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,history=QoSHistoryPolicy.KEEP_LAST,durability=QoSDurabilityPolicy.VOLATILE,depth=1)
pub=n.create_publisher(MotionCommand,'topic_control_signal',qos)
g=n.create_client(GetEntityState,'/gazebo/get_entity_state');g.wait_for_service(timeout_sec=10)
st=n.create_client(SetEntityState,'/gazebo/set_entity_state');st.wait_for_service(timeout_sec=10)
r=SetEntityState.Request();r.state.name='ego_vehicle'
r.state.pose.position.x=sx;r.state.pose.position.y=sy;r.state.pose.position.z=0.05
r.state.pose.orientation.z=math.sin(syaw/2);r.state.pose.orientation.w=math.cos(syaw/2);r.state.reference_frame='world'
rclpy.spin_until_future_complete(n,st.call_async(r),timeout_sec=3);time.sleep(1)
print(f'출발 세팅 ({sx:.1f},{sy:.1f},{math.degrees(syaw):.0f}°), 명령 {len(seq)}개 재생...')
for row in seq:
    m=MotionCommand();m.steering=int(row['steer']);m.left_speed=int(row['l']);m.right_speed=int(row['r']);pub.publish(m)
    time.sleep(0.1)
m=MotionCommand();pub.publish(m)  # 정지
time.sleep(1)
f=g.call_async(GetEntityState.Request(name='ego_vehicle'));rclpy.spin_until_future_complete(n,f,timeout_sec=3)
p=f.result().state.pose.position;q=f.result().state.pose.orientation
yaw=math.degrees(math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z)))
SLOT3=(-3.15,1.91)
print(f'재생 최종: ({p.x:.2f},{p.y:.2f},{yaw:.0f}°), 슬롯3 GT까지 {math.hypot(p.x-SLOT3[0],p.y-SLOT3[1]):.2f}m')
print('→ '+('★주차 재현 성공★' if math.hypot(p.x-SLOT3[0],p.y-SLOT3[1])<1.2 else '재현 오차 큼(비결정성?)'))
n.destroy_node();rclpy.shutdown()
