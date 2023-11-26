import numpy as np
from Model_V1 import Schedule

class ScheduleOptimizer:
    def __init__(self, schedule):
        self.schedule = schedule

    def optimize(self, generations):
        best_fitness = -np.inf
        best_class_teacher_matrix = None

        for _ in range(generations):
            class_teacher_matrix = self.generate_random_matrix()
            self.schedule.class_teacher_matrix = class_teacher_matrix
            fitness = self.schedule.Fitness_Function()

            if fitness > best_fitness:
                best_fitness = fitness
                best_class_teacher_matrix = class_teacher_matrix

        self.schedule.class_teacher_matrix = best_class_teacher_matrix
        return best_class_teacher_matrix

    def generate_random_matrix(self):
        total_num_classes = self.schedule.total_num_classes
        num_teachers = self.schedule.num_teachers
        class_teacher_matrix = np.zeros((total_num_classes, num_teachers), dtype=int)

        for class_num in range(total_num_classes):
            teacher_num = np.random.randint(num_teachers)
            class_teacher_matrix[class_num, teacher_num] = 1

        return class_teacher_matrix

# Usage example
schedule1 = Schedule(3, 3, 2, 3)
schedule1.set_class_course(1, 1, 1)
schedule1.set_class_course(2, 2, 1)
schedule1.set_class_course(3, 3, 1)



schedule1.set_teacher_course_interest(1, 1, 5)
schedule1.set_teacher_course_interest(1, 2, 9)
schedule1.set_teacher_course_interest(1, 3, 8)

schedule1.set_teacher_course_interest(2, 1, 5)
schedule1.set_teacher_course_interest(2, 2, 9)
schedule1.set_teacher_course_interest(2, 3, 8)

schedule1.set_teacher_course_quality(1, 1, 10)
schedule1.set_teacher_course_quality(1, 2, 9)
schedule1.set_teacher_course_quality(1, 3, 8)

schedule1.set_teacher_course_quality(2, 1, 10)
schedule1.set_teacher_course_quality(2, 2, 9)
schedule1.set_teacher_course_quality(2, 3, 8)


schedule1.set_teacher_time_preference(1, 1, 10)
schedule1.set_teacher_time_preference(1, 2, 5)
schedule1.set_teacher_time_preference(1, 3, 5)

schedule1.set_teacher_time_preference(2, 1, 8)
schedule1.set_teacher_time_preference(2, 2, 5)
schedule1.set_teacher_time_preference(2, 3, 5)



schedule1.set_teacher_desired_classes(1, 1)
schedule1.set_teacher_desired_classes(2, 1)


schedule1.set_class_time(1, 1, 1)
schedule1.set_class_time(2, 2, 1)


schedule1.set_class_teacher(1, 1, 1)
schedule1.set_class_teacher(1, 2, 1)

schedule1.calculate_teacher_time_priority()
schedule1.calculate_teacher_class_interest()
schedule1.calculate_teacher_class_quality()


schedule1.display_schedule()



optimizer = ScheduleOptimizer(schedule1)
optimized_class_teacher_matrix = optimizer.optimize(generations=10000)

schedule1.class_teacher_matrix = optimized_class_teacher_matrix
schedule1.display_schedule()
print(schedule1.Fitness_Function())


