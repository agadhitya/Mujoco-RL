import csv
from datetime import datetime
import matplotlib.pyplot as plt

x = []
y = []
action=[]
object_t = []
action_t = []
Zero =datetime.strptime("00:00:00","%H:%M:%S")

def time_seconds_conversion(time):
    # h, m, s = (int(x) for x in time.split(":"))
    # return 3600 * h + 60 * m + s
    global Zero
    return (datetime.strptime(time,"%H:%M:%S")-Zero).seconds

with open('/home/gsathyanarayanan/catkin_ws/object_position.csv','r') as csvfile:
    # global Zero

    plots = csv.reader(csvfile, delimiter=',')
    n=0

    for row in plots:

        n=n+1
        if(n==1):
            continue
        if(n==2):
            Zero = datetime.strptime(row[0][11:19],"%H:%M:%S")

        x.append(float(row[1]))
        y.append(float(row[2]))
        time= (row[0][11:19])
        object_t.append(time_seconds_conversion(time))

with open('/home/gsathyanarayanan/catkin_ws/Action.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    n = 0

    for row in plots:

        n = n + 1
        if (n == 1):
            continue
        if (n == 2):
            Zero = datetime.strptime(row[0][11:19], "%H:%M:%S")


        time = (row[0][11:19])
        action_t.append(time_seconds_conversion(time))
        action.append(int(row[1]))

    plt.plot(x,y,'b',zorder=6)

    a=0
    X=[]
    Y=[]
    for i in range(len(object_t)):
        print action_t[a],object_t[i],action[a]
        if (object_t[i]<=action_t[a] or a==(len(action_t)-1)):

            if((action[a])==5 or (action[a] ==4)):
                print x[i],y[i]
                # print action[a]
                X.append(x[i])
                Y.append(y[i])
                #plt.plot(x[i],y[i],'r.')
            else:
                continue
        else:
            a=a+1
            if a==len(action_t):
                a=len(action_t)-1
plt.show()
#plt.plot(X,Y, 'r',zorder=5)