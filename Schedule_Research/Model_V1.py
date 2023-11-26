import numpy as np
import random
import math 
class Schedule:
    def __init__(self, num_courses, num_time_slots, num_teachers, total_num_classes):
        self.num_courses = num_courses
        self.num_time_slots = num_time_slots
        self.num_teachers = num_teachers
        self.total_num_classes = total_num_classes
        self.class_course_matrix = np.zeros((total_num_classes, num_courses), dtype=int)
        self.class_time_matrix = np.zeros((total_num_classes, num_time_slots), dtype=int)
        self.teacher_course_interest_matrix = np.zeros((num_teachers, num_courses), dtype=float)
        self.teacher_class_interest_matrix = np.zeros((num_teachers, total_num_classes), dtype=float)
        self.teacher_course_quality_matrix = np.zeros((num_teachers, num_courses), dtype=float)
        self.teacher_class_quality_matrix = np.zeros((num_teachers, total_num_classes), dtype=float)
        self.teacher_time_preference_matrix = np.zeros((num_teachers, num_time_slots), dtype=float)
        self.teacher_time_priority_matrix = np.zeros((num_teachers, num_time_slots), dtype=float)
        self.teacher_desired_classes = np.zeros(num_teachers, dtype=int)
        self.teacher_min_classes = np.ones(num_teachers, dtype=int)
        self.class_bonus=(math.ceil((self.total_num_classes-self.num_teachers)))+100
        self.teacher_max_classes = np.ones(num_teachers, dtype=int)* self.class_bonus
        self.class_teacher_matrix = np.zeros((total_num_classes, num_teachers), dtype=int)
        self.epsilon=1
        # Agent
       

    # Agent function
    def Generate_Optimazation_Schedule(self):
        class_observation_space = list(range(1, self.total_num_classes+1))
        teacher_action_space = list(range(1, self.num_teachers+1))
        reward_fitness = 0
        solution = None
        for i in range(1000):
            for class_index in class_observation_space:
                teacher_choice = random.choice(teacher_action_space)
                while(True):
                   if self.set_class_teacher(class_index, teacher_choice, 1)==True:
                       print('Sucess Set: '+str(class_index)+'-'+str(teacher_choice))
                       break
                   else:
                       pass
                       print('Fail Set: '+str(class_index)+'-'+str(teacher_choice))
                   teacher_choice = random.choice(teacher_action_space)
            if self.Fitness_Function()>reward_fitness :
                reward_fitness= self.Fitness_Function()
                solution= self.class_teacher_matrix
            self.class_teacher_matrix = np.zeros((self.total_num_classes, self.num_teachers), dtype=int)
        print(reward_fitness)
        print(solution)

    def Generate_Optimazation_Schedule_V2(self):
        class_observation_space = list(range(1, self.total_num_classes+1))
        teacher_action_space = list(range(1, self.num_teachers+1))
        reward_fitness = 0
        solution = None
        Q= np.zeros((self.total_num_classes, self.num_teachers), dtype=float)
        for i in range(2000):
            for class_index in class_observation_space:
                # teacher_choice = random.choice(teacher_action_space)
                teacher_choice = self.action(Q,class_index,teacher_action_space,i,1000)
                counttry = 0
                while(True):
                   if self.set_class_teacher(class_index, teacher_choice, 1)==True:
                       Q=self.train(Q,class_index,teacher_choice,self.Fitness_Function(),class_index+1)
                       break
                   else:
                       if counttry >=1000:
                        for index, class_index in enumerate(self.class_teacher_matrix):
                            Q=self.train(Q,index+1,np.argmax(class_index)+1,-1,index+1)
                        break
                       counttry+=1
                       Q=self.train(Q,class_index,teacher_choice,-1,class_index+1)
                #    teacher_choice = random.choice(teacher_action_space)
                   teacher_choice = self.action(Q,class_index,teacher_action_space,i,1000)
            
            if self.Fitness_Function()>reward_fitness :
                reward_fitness= self.Fitness_Function()
                solution= self.class_teacher_matrix
                for index, class_index in enumerate(solution):
                    Q=self.train(Q,index+1,np.argmax(class_index)+1,1,index+1)

            self.class_teacher_matrix = np.zeros((self.total_num_classes, self.num_teachers), dtype=int)
        print(reward_fitness)
        print(solution)
        # print(Q)
        print(self.epsilon)
        # Final lap
        self.class_teacher_matrix = np.zeros((self.total_num_classes, self.num_teachers), dtype=int)
        for class_index in class_observation_space:
            if self.set_class_teacher(class_index, np.argmax(Q[class_index-1])+1, 1)==True:
                pass
            else:
                raise ValueError("Bad Solution ")
        print("Final result,Fitness:")
        print((self.class_teacher_matrix,self.Fitness_Function()))

    def action(self,Q,class_index,teacher_action_space,i,N):
        if np.random.rand() > self.epsilon:
             return  np.argmax(Q[class_index-1])+1
        else:
            self.epsilon = 1 - ((i/N)**20) 
            return random.choice(teacher_action_space)

    def train(self,Q, state, action, reward, next_state):
        state=state-1
        action=action-1
        next_state=next_state-1
        if(next_state>=self.total_num_classes):
            next_state=self.total_num_classes-1
        Q[state, action] = Q[state, action] + 0.8 * (reward + 0.8 * np.max(Q[next_state]) - Q[state, action])
        return Q
    
   
    def set_class_course(self, class_num, course_num, value):
        if class_num <= self.total_num_classes and course_num <= self.num_courses and class_num > 0 and course_num > 0:
            if np.count_nonzero(self.class_course_matrix.T[course_num-1]) != 0:
                raise ValueError("This course already belong to another class (Set another course for this class)")
            self.class_course_matrix[class_num-1, course_num-1] = value
        else:
            raise ValueError("Invalid class or course number.")

    def set_class_time(self, class_num, time_num, value):
        if class_num <= self.total_num_classes and time_num <= self.num_time_slots and class_num > 0 and time_num > 0:
            self.class_time_matrix[class_num-1, time_num-1] = value
            for time_index in self.class_time_matrix.T:
                if int(np.count_nonzero(time_index)) > int(self.num_teachers):
                    raise ValueError("This time slot have at lest (num teacher): "+str(np.count_nonzero(time_index)))
        else:
            raise ValueError("Invalid class or time number.")

    def set_teacher_course_interest(self, teacher_num, course_num, value):
        if teacher_num <= self.num_teachers and course_num <= self.num_courses and 0 <= value <= 10 and teacher_num > 0 and course_num > 0:
            self.teacher_course_interest_matrix[teacher_num-1, course_num-1] = value
        else:
            raise ValueError("Invalid teacher, course, or interest value.")

    def calculate_teacher_class_interest(self):
        self.teacher_class_interest_matrix = np.matmul(self.teacher_course_interest_matrix, self.class_course_matrix.T)

    def set_teacher_course_quality(self, teacher_num, course_num, value):
        if teacher_num <= self.num_teachers and course_num <= self.num_courses and 0 <= value <= 10 and teacher_num > 0 and course_num > 0:
            self.teacher_course_quality_matrix[teacher_num-1, course_num-1] = value
        else:
            raise ValueError("Invalid teacher, course, or quality value.")

    def set_teacher_class_quality(self, teacher_num, class_num, value):
        if teacher_num <= self.num_teachers and class_num <= self.total_num_classes and 0 <= value <= 10 and teacher_num > 0 and class_num > 0:
            self.teacher_class_quality_matrix[teacher_num-1, class_num-1] = value
        else:
            raise ValueError("Invalid teacher, class, or quality value.")

    def calculate_teacher_class_quality(self):
        self.teacher_class_quality_matrix = np.matmul(self.teacher_course_quality_matrix, self.class_course_matrix.T)

    def set_teacher_time_preference(self, teacher_num, time_num, value):
        if teacher_num <= self.num_teachers and time_num <= self.num_time_slots and 0 <= value <= 10 and teacher_num > 0 and time_num > 0:
            self.teacher_time_preference_matrix[teacher_num-1, time_num-1] = value
        else:
            raise ValueError("Invalid teacher, time, or preference value.")

    def calculate_teacher_time_priority(self):
        self.teacher_time_priority_matrix = np.matmul(self.teacher_time_preference_matrix, self.class_time_matrix.T)

    def set_teacher_desired_classes(self, teacher_num, value):
        if teacher_num <= self.num_teachers and 0 <= value <= 10 and teacher_num > 0 and value <= self.class_bonus:
            if value > self.teacher_max_classes[teacher_num-1] or value < self.teacher_min_classes[teacher_num-1] :
                 raise ValueError("Invalid teacher or desired classes value. mus be in range max and min")
            self.teacher_desired_classes[teacher_num-1] = value
        else:
            raise ValueError("Invalid teacher or desired classes value.")

    def set_teacher_min_classes(self, teacher_num, value):

        if teacher_num <= self.num_teachers and 0 <= value <= 10 and teacher_num > 0:
            self.teacher_min_classes[teacher_num-1] = value
        else:
            raise ValueError("Invalid teacher or minimum classes value.")

    def set_teacher_max_classes(self, teacher_num, value):
        if teacher_num <= self.num_teachers and 0 <= value <= 10 and teacher_num > 0:
            self.teacher_max_classes[teacher_num-1] = value
        else:
            raise ValueError("Invalid teacher or maximum classes value.")
    def set_class_teacher(self, class_num, teacher_num, value):
        try:
            if class_num <= self.total_num_classes and teacher_num <= self.num_teachers and class_num > 0 and teacher_num > 0:
                
                if self.valid_time_slot(class_num,teacher_num) == False:
                    return False
                if self.valid_onyone_teacher( class_num)== False:
                    return False
                if self.valid_interest_subject_teacher(class_num,teacher_num)== False:
                    return False
                if self.valid_quality_subject_teacher(class_num,teacher_num)== False:
                    return False
                if self.valid_timeslot_preference_teacher(class_num,teacher_num)== False:
                    return False

                self.class_teacher_matrix[class_num-1, teacher_num-1] = value
                return True
            else:
                return False
                raise ValueError("Invalid class or teacher number.")
        except IndexError as e:
            print(e)
            return False
    def valid_time_slot(self, class_num, teacher_num):
        for x in self.get_class_time_slots(teacher_num):
            if self.get_class_time_slot(class_num)[1] == x[1]:
                return False
                raise ValueError("Confict time slot same teacher")
    
    def valid_onyone_teacher(self, class_num):
        if np.count_nonzero(self.class_teacher_matrix[class_num-1]) != 0:
            return False
            raise ValueError("This class belong to another teacher")
    
    def valid_interest_subject_teacher(self, class_num, teacher_num):
        if  self.teacher_course_interest_matrix[teacher_num-1][self.get_course_by_class(class_num)[1]-1] <= 0 :
            return False
            raise ValueError("Teacher no interest in this class")
    
    def valid_quality_subject_teacher(self, class_num, teacher_num):
        if  self.teacher_course_quality_matrix[teacher_num-1][self.get_course_by_class(class_num)[1]-1] <= 0 :
            return False
            raise ValueError("Teacher low quality  in this class")
    
    def valid_timeslot_preference_teacher(self, class_num, teacher_num):
        if  self.teacher_time_priority_matrix[teacher_num-1][self.get_course_by_class(class_num)[1]-1] <= 0 :
            return False
            raise ValueError("Teacher low time Priority in this class")
        
    def valid_MAX_MIN_CLASS(self, teacher_num):
        if  np.count_nonzero(self.class_teacher_matrix.T[teacher_num-1])+1 > self.teacher_max_classes[teacher_num-1] :
            return False
            raise ValueError("this teacher has reach to the max class can take !")
        # if  np.count_nonzero(self.class_teacher_matrix.T[teacher_num-1])+1 > self.teacher_min_classes[teacher_num-1] :
        #     raise ValueError("this teacher not meet the minium class to take !")

    def get_class_time_slots(self, teacher_num):
        if teacher_num <= self.num_teachers and teacher_num > 0:
            classes_time_slots = []
            for class_num in range(1, self.total_num_classes + 1):
                if self.class_teacher_matrix[class_num - 1, teacher_num - 1] == 1:
                    time_slots = np.where(self.class_time_matrix[class_num - 1] == 1)[0]
                    for time_slot in time_slots:
                        classes_time_slots.append((class_num, (time_slot+1)))
            return classes_time_slots
        else:
            raise ValueError("Invalid teacher number.")
        
    def get_class_time_slot(self, class_num):
        try:
            class_num,(np.where(self.class_time_matrix[class_num - 1] == 1)[0][0])+1
        except:
            raise ValueError("Check set time slot to all class .")
        return(class_num,(np.where(self.class_time_matrix[class_num - 1] == 1)[0][0])+1)

    def payofffunctionP0(self):
        payoff = np.sum( self.class_teacher_matrix *  self.teacher_class_quality_matrix.T)
        return payoff
    
    def payofffunctionPi(self):
        lsi = np.sum( self.teacher_class_interest_matrix *  self.class_teacher_matrix.T)
        lti = np.sum( self.teacher_time_priority_matrix *  self.class_teacher_matrix.T)
        lai =10-abs(self.teacher_desired_classes- np.sum(self.class_teacher_matrix.T))
        return 0.3*lsi+0.3*lti+0.3*lai
   
    def Fitness_Function(self):
        p0 = self.payofffunctionP0()
        pi = self.payofffunctionPi()
        fitness = 0.5 * p0 + 0.5 * pi
        return np.sum(fitness)

    def get_course_by_class(self,class_num):
        try:
            class_num,(np.where(self.class_course_matrix[class_num - 1] == 1)[0][0])+1
        except:
            raise ValueError("Check set cousre to all class .")
        return class_num,(np.where(self.class_course_matrix[class_num - 1] == 1)[0][0])+1    

    def Apply_schedule(self):
        self.calculate_teacher_time_priority()
        self.calculate_teacher_class_interest()
        self.calculate_teacher_class_quality()       

    def display_schedule(self):
        print("Number of Courses:", self.num_courses)
        print("Number of Time Slots:", self.num_time_slots)
        print("Number of Teachers:", self.num_teachers)
        print("Total Number of Classes:", self.total_num_classes)
        print("Class-Course Matrix:")
        print(self.class_course_matrix)
        print("Class-Time Matrix:")
        print(self.class_time_matrix)
        print("Teacher-Course Interest Matrix:")
        print(self.teacher_course_interest_matrix)
        print("Teacher-Class Interest Matrix:")
        print(self.teacher_class_interest_matrix)
        print("Teacher Course Quality Matrix:")
        print(self.teacher_course_quality_matrix)
        print("Teacher Class Quality Matrix:")
        print(self.teacher_class_quality_matrix)
        print("Teacher-Time Preference Matrix:")
        print(self.teacher_time_preference_matrix)
        print("Teacher-Time Priority Matrix:")
        print(self.teacher_time_priority_matrix)
        print("Teacher Desired Classes:")
        print(self.teacher_desired_classes)
        print("Teacher Min Classes:")
        print(self.teacher_min_classes)
        print("Teacher Max Classes:")
        print(self.teacher_max_classes)
        print("Class-Teacher Matrix:")
        print(self.class_teacher_matrix)
        print("Teacher-Class Matrix :")
        print(self.class_teacher_matrix.T)
        print("P0,Pi,finess :")
        print(self.payofffunctionP0())
        print(self.payofffunctionPi())
        print(self.Fitness_Function())




# (self, num_courses, num_time_slots, num_teachers, total_num_classes)
schedule1 = Schedule(8, 6, 3, 8)



schedule1.set_class_course(1, 1, 1)
schedule1.set_class_course(2, 2, 1)
schedule1.set_class_course(3, 3, 1)
schedule1.set_class_course(4, 4, 1)
schedule1.set_class_course(5, 5, 1)
schedule1.set_class_course(6, 6, 1)
schedule1.set_class_course(7, 7, 1)
schedule1.set_class_course(8, 8, 1)

schedule1.set_class_time(1, 1, 1)
schedule1.set_class_time(2, 2, 1)
schedule1.set_class_time(3, 3, 1)
schedule1.set_class_time(4, 1, 1)
schedule1.set_class_time(5, 2, 1)
schedule1.set_class_time(6, 3, 1)
schedule1.set_class_time(7, 4, 1)
schedule1.set_class_time(8, 5, 1)



schedule1.set_teacher_course_interest(1, 1, 10)
schedule1.set_teacher_course_interest(1, 2, 2)
schedule1.set_teacher_course_interest(1, 3, 8)
schedule1.set_teacher_course_interest(1, 4, 5)
schedule1.set_teacher_course_interest(1, 4, 7)
schedule1.set_teacher_course_interest(1, 5, 6)
schedule1.set_teacher_course_interest(1, 6, 7)
schedule1.set_teacher_course_interest(1, 7, 4)
schedule1.set_teacher_course_interest(1, 8, 1)
schedule1.set_teacher_course_interest(2, 1, 9)
schedule1.set_teacher_course_interest(2, 2, 6)
schedule1.set_teacher_course_interest(2, 3, 10)
schedule1.set_teacher_course_interest(2, 4, 4)
schedule1.set_teacher_course_interest(2, 5, 1)
schedule1.set_teacher_course_interest(2, 6, 2)
schedule1.set_teacher_course_interest(2, 7, 7)
schedule1.set_teacher_course_interest(2, 8, 5)
schedule1.set_teacher_course_interest(3, 1, 9)
schedule1.set_teacher_course_interest(3, 2, 6)
schedule1.set_teacher_course_interest(3, 3, 10)
schedule1.set_teacher_course_interest(3, 4, 4)
schedule1.set_teacher_course_interest(3, 5, 1)
schedule1.set_teacher_course_interest(3, 6, 2)
schedule1.set_teacher_course_interest(3, 7, 7)
schedule1.set_teacher_course_interest(3, 8, 5)


schedule1.set_teacher_course_quality(1, 1, 5)
schedule1.set_teacher_course_quality(1, 2, 4)
schedule1.set_teacher_course_quality(1, 3, 8)
schedule1.set_teacher_course_quality(1, 4, 6)
schedule1.set_teacher_course_quality(1, 5, 9)
schedule1.set_teacher_course_quality(1, 6, 10)
schedule1.set_teacher_course_quality(1, 7, 8)
schedule1.set_teacher_course_quality(1, 8, 9)
schedule1.set_teacher_course_quality(2, 1, 4)
schedule1.set_teacher_course_quality(2, 2, 8)
schedule1.set_teacher_course_quality(2, 3, 9)
schedule1.set_teacher_course_quality(2, 4, 6)
schedule1.set_teacher_course_quality(2, 5, 10)
schedule1.set_teacher_course_quality(2, 6, 5)
schedule1.set_teacher_course_quality(2, 7, 7)
schedule1.set_teacher_course_quality(2, 8, 8)
schedule1.set_teacher_course_quality(3, 1, 4)
schedule1.set_teacher_course_quality(3, 2, 8)
schedule1.set_teacher_course_quality(3, 3, 9)
schedule1.set_teacher_course_quality(3, 4, 6)
schedule1.set_teacher_course_quality(3, 5, 10)
schedule1.set_teacher_course_quality(3, 6, 5)
schedule1.set_teacher_course_quality(3, 7, 7)
schedule1.set_teacher_course_quality(3, 8, 8)

schedule1.set_teacher_time_preference(1, 1, 8)
schedule1.set_teacher_time_preference(1, 2, 9)
schedule1.set_teacher_time_preference(1, 3, 7)
schedule1.set_teacher_time_preference(1, 4, 6)
schedule1.set_teacher_time_preference(1, 5, 7)
schedule1.set_teacher_time_preference(1, 6, 9)
schedule1.set_teacher_time_preference(2, 1, 10)
schedule1.set_teacher_time_preference(2, 2, 5)
schedule1.set_teacher_time_preference(2, 3, 6)
schedule1.set_teacher_time_preference(2, 4, 8)
schedule1.set_teacher_time_preference(2, 5, 4)
schedule1.set_teacher_time_preference(2, 6, 3)
schedule1.set_teacher_time_preference(3, 1, 10)
schedule1.set_teacher_time_preference(3, 2, 5)
schedule1.set_teacher_time_preference(3, 3, 6)
schedule1.set_teacher_time_preference(3, 4, 8)
schedule1.set_teacher_time_preference(3, 5, 4)
schedule1.set_teacher_time_preference(3, 6, 3)


schedule1.set_teacher_desired_classes(1, 4)
schedule1.set_teacher_desired_classes(2, 4)
schedule1.set_teacher_desired_classes(3, 4)




schedule1.Apply_schedule()
# schedule1.set_class_teacher(1, 1, 1)
# schedule1.set_class_teacher(2, 2, 1)
# schedule1.set_class_teacher(3, 2, 1)

# schedule1.display_schedule()
# schedule1.Generate_Optimazation_Schedule()
schedule1.Generate_Optimazation_Schedule_V2()