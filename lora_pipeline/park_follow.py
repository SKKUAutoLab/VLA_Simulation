"""녹화 pose경로 폐루프 추종 — 사람 주차 시연 경로를 따라감(전진/후진 구간 반영)."""
import csv,math,rclpy,time
from rclpy.node import Node
from interfaces_pkg.msg import MotionCommand
from gazebo_msgs.srv import GetEntityState, SetEntityState
from rclpy.qos import QoSProfile,QoSReliabilityPolicy,QoSHistoryPolicy,QoSDurabilityPolicy
def norm(a): return math.atan2(math.sin(a),math.cos(a))
rows=list(csv.DictReader(open('/tmp/park_demo.csv')))
def nz(r): return int(r['steer'])!=0 or int(r['l'])!=0 or int(r['r'])!=0
drv=[i for i,r in enumerate(rows) if nz(r)]
seq=rows[drv[0]:drv[-1]+1]
# 경로점: (x,y,dir)  dir=+1전진/-1후진 (녹화 l부호)
PATH=[(float(r['x']),float(r['y']), -1 if int(r['l'])<0 else 1) for r in seq]
sx,sy,syaw=float(seq[0]['x']),float(seq[0]['y']),float(seq[0]['yaw'])
GOAL=(float(seq[-1]['x']),float(seq[-1]['y']))
rclpy.init();n=Node('fol')
qos=QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,history=QoSHistoryPolicy.KEEP_LAST,durability=QoSDurabilityPolicy.VOLATILE,depth=1)
pub=n.create_publisher(MotionCommand,'topic_control_signal',qos)
g=n.create_client(GetEntityState,'/gazebo/get_entity_state');g.wait_for_service(timeout_sec=10)
sc=n.create_client(SetEntityState,'/gazebo/set_entity_state');sc.wait_for_service(timeout_sec=10)
def pose():
    f=g.call_async(GetEntityState.Request(name='ego_vehicle'));rclpy.spin_until_future_complete(n,f,timeout_sec=2)
    if not f.result(): return None
    p=f.result().state.pose.position;q=f.result().state.pose.orientation
    return p.x,p.y,math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
def send(s,sp):
    m=MotionCommand();m.steering=int(s);m.left_speed=int(sp);m.right_speed=int(sp);pub.publish(m)
# 출발 세팅
r=SetEntityState.Request();r.state.name='ego_vehicle'
r.state.pose.position.x=sx;r.state.pose.position.y=sy;r.state.pose.position.z=0.05
r.state.pose.orientation.z=math.sin(syaw/2);r.state.pose.orientation.w=math.cos(syaw/2);r.state.reference_frame='world'
rclpy.spin_until_future_complete(n,sc.call_async(r),timeout_sec=3);time.sleep(1)
print(f'출발({sx:.1f},{sy:.1f}) 경로{len(PATH)}점 추종...')
idx=0; N=len(PATH)
for step in range(1500):
    pr=pose()
    if pr is None: continue
    x,y,yaw=pr; fwd=yaw-math.pi/2
    # 진행: 현재위치서 가장가까운 경로점 이후로 idx 전진
    while idx<N-1 and math.hypot(PATH[idx][0]-x,PATH[idx][1]-y)<0.6: idx+=1
    if idx>=N-1 and math.hypot(GOAL[0]-x,GOAL[1]-y)<0.4:
        send(0,0);print(f'✅ 추종완료 step{step} ({x:.2f},{y:.2f}) goal까지{math.hypot(GOAL[0]-x,GOAL[1]-y):.2f}m');break
    # 룩어헤드 점(8점 앞)
    li=min(idx+8,N-1); lx,ly,d=PATH[li]
    los=math.atan2(ly-y,lx-x)
    if d>0:  # 전진
        he=norm(los-fwd); sp=40
    else:    # 후진: 후미를 룩어헤드로 → 기준 반전
        he=norm((los+math.pi)-fwd); sp=-40
    st=max(-7,min(7, he*9))
    send(st,sp); time.sleep(0.1)
else:
    pr=pose();send(0,0);print(f'⏱미완 ({pr[0]:.2f},{pr[1]:.2f}) goal까지{math.hypot(GOAL[0]-pr[0],GOAL[1]-pr[1]):.2f}m')
fp=pose();SLOT3=(-3.15,1.91)
print(f'최종 슬롯3까지 {math.hypot(fp[0]-SLOT3[0],fp[1]-SLOT3[1]):.2f}m')
n.destroy_node();rclpy.shutdown()
