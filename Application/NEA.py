from ursina import*
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import time
from ursina.prefabs import trail_renderer
import numpy as np
import math
import sympy as smp
from scipy.spatial.transform import Rotation as R
vec = [1,1,1]

rotation_degrees = 90
rotation_radians = np.radians(rotation_degrees)
rotation_axis = np.array([0, 0, 1])

rotation_vector = rotation_radians * rotation_axis
rotation = R.from_rotvec(rotation_vector)
rotated_vec = rotation.apply(vec)


app = Ursina()
window.title = 'Satellite Prototype'
window.borderless = False
window.exit_button.visible = False
window.fps_counter.enabled = False

t = 0
momentum1 = [0,0,0]
momentum2 = [0,30,0]

AU = (149.6e6 * 1000)
SCALE = 250 / AU
dt = 3600 * 24


class Object(Entity):

    def __init__(self,mass,radius,model,texture,_position,velocity,scale):
        super().__init__(model=model, texture=texture)
        self.m = 0
        self.scale = scale
        self.mass = mass
        self.radius = radius
        self._position = _position
        self._origin = [0,0,0]
        self.static = False
        self.velocity = velocity
        self._acceleration = [[0.0, 0.0, 0.0]]
        self._vnew = [0.0, 0.0, 0.0]
        self._partialPos = [0.0, 0.0, 0.0]
        self.G = 1

    def UpdatePartialPos(self, index, dt):
        self._partialPos[0] = self._position[index][0] + self._velocity[index][0] * dt / 2.0
        self._partialPos[1] = self._position[index][1] + self._velocity[index][1] * dt / 2.0
        self._partialPos[2] = self._position[index][2] + self._velocity[index][2] * dt / 2.0

        self._vnew = self._velocity[index].copy()


    def GravitationalForce(self):
        G = 1
        r_vec = np.subtract(self._position,self._origin)
        self.r_mag = np.linalg.norm(r_vec)

        f = (G *self.mass*10000) / np.power(self.r_mag, 2)


        return f

    def V_vec(self):

        theta = 0
        theta_vec = Vec3(-math.sin(theta),0,math.cos(theta))
        print(theta_vec)
        d0dt = 0.021
        V_vec = self.r_mag * d0dt * theta_vec
        theta += d0dt
        print(V_vec)
        return V_vec

E_scale = 5.972e24/6500
S_scale = 6500/5.972e24
E = Object(mass = 5.972e24, radius = 695508e3, model='sphere',texture="Earth-map", _position = [0,0,0],velocity= [0,0,0],scale = E_scale)
E.position = E._position
E.scale = 10
E.static = False

G = 6.67e-11

v2 = G*E.mass/(E.radius+35786e3)
v = math.sqrt(v2)/3000

S = Object(mass = 6500, radius = 2439.4e3, model='sphere', texture ="texty", _position = [100,0,0],velocity= [0,0,v],scale = S_scale)
S.position = S.position
S.staic = False
S.scale = S.radius/E.radius*100
S._maxPoint = 366


class TrailRenderer(Entity):
    def __init__(self, thickness=10, color=color.white, end_color=color.clear, length=500, **kwargs):
        super().__init__(**kwargs)
        self.renderer = Entity(
            model = Mesh(
            vertices=[self.world_position for i in range(length)],
            colors=[lerp(end_color, color, i/length*2) for i in range(length)],
            mode='line',
            thickness=thickness,
            static=False
            )
        )
        self._t = 0
        self.update_step = .025


    def update(self):
        self._t += time.dt
        if self._t >= self.update_step:
            self._t = 0
            self.renderer.model.vertices.pop(0)
            self.renderer.model.vertices.append(self.world_position)
            self.renderer.model.generate()

    def on_destroy(self):
        destroy(self.renderer)




if __name__ == '__main__':
    player = S
    trail_renderer = TrailRenderer(parent=player, thickness=1, color=color.yellow, length=10)

    pivot = Entity(parent=S)
    trail_renderer = TrailRenderer(parent=pivot, x=.1, thickness=20, color=color.orange)
    trail_renderer = TrailRenderer(parent=pivot, y=1, thickness=20, color=color.orange)
    trail_renderer = TrailRenderer(parent=pivot, thickness=2, color=color.orange, alpha=.5, position=(.4,.8))
    trail_renderer = TrailRenderer(parent=pivot, thickness=2, color=color.orange, alpha=.5, position=(-.5,.7))

    def update():
        player.position = lerp(player.position, mouse.position*10, time.dt*4)

        if pivot:
            pivot.rotation_z -= 3
            pivot.rotation_x -= 2

    def input(key):
        if key == 'space':
            destroy(pivot)




def update():
    global t

    r_vec = np.subtract(S._position, E._position)
    distance = np.linalg.norm(r_vec)

    direction_vector = [S._position[0] - E._position[0], S._position[1] - E._position[1],
                            S._position[2] - E._position[2]]

    A = S.GravitationalForce()

    xv = (A / S.mass * time.dt / 20) / distance * -direction_vector[0]
    yv = (A / S.mass * time.dt / 20) / distance * -direction_vector[1]
    zv = (A / S.mass * time.dt / 20) / distance * -direction_vector[2]
    S.velocity[0] += xv
    S.velocity[1] += yv
    S.velocity[2] += zv
    print(S.velocity[2])
    print(S._position[2])

    new_position = [S._position[0]+S.velocity[0], S._position[1]+S.velocity[1], S._position[2]+S.velocity[2]]
    S._position = new_position

    S.position = new_position
    print(S._position)



speed = 5.1
Sky(texture = 'cry.png')
index = 0
pause = False
EditorCamera(x=0,y=0,z=-50)
app.run()
