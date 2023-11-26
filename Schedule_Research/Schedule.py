import numpy as np

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
        self.teacher_min_classes = np.zeros(num_teachers, dtype=int)
        self.teacher_max_classes = np.zeros(num_teachers, dtype=int)

    def set_class_course(self, class_num, course_num, value):
        if class_num <= self.total_num_classes and course_num <= self.num_courses and class_num > 0 and course_num > 0:
            self.class_course_matrix[class_num-1, course_num-1] = value
        else:
            raise ValueError("Invalid class or course number.")

    def set_class_time(self, class_num, time_num, value):
        if class_num <= self.total_num_classes and time_num <= self.num_time_slots and class_num > 0 and time_num > 0:
            self.class_time_matrix[class_num-1, time_num-1] = value
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
        if teacher_num <= self.num_teachers and course_num <= self.num_courses and 0 < value < 10 and teacher_num > 0 and course_num > 0:
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
        if teacher_num <= self.num_teachers and 0 <= value <= 10 and teacher_num > 0:
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


# Sử dụng lớp Schedule
schedule1 = Schedule(3, 3, 2, 3)
schedule1.set_class_course(1, 1, 1)
schedule1.set_class_course(2, 2, 1)
schedule1.set_class_course(3, 1, 1)
schedule1.set_class_time(1, 1, 1)
schedule1.set_class_time(2, 2, 1)
schedule1.set_class_time(3, 3, 1)
schedule1.set_teacher_course_interest(1, 1, 8)
schedule1.set_teacher_course_interest(1, 2, 9)
schedule1.set_teacher_course_interest(2, 1, 7)
schedule1.set_teacher_course_interest(2, 2, 6)
schedule1.set_teacher_course_quality(1, 1, 9.5)
schedule1.set_teacher_course_quality(1, 2, 8.2)
schedule1.set_teacher_course_quality(2, 1, 7.8)
schedule1.set_teacher_course_quality(2, 2, 9.1)
schedule1.calculate_teacher_class_interest()
schedule1.calculate_teacher_class_quality()
schedule1.set_teacher_time_preference(1, 1, 7.8)
schedule1.set_teacher_time_preference(2, 2, 9.1)
schedule1.calculate_teacher_time_priority()
schedule1.set_teacher_desired_classes(1, 2)
schedule1.set_teacher_desired_classes(2, 3)
schedule1.set_teacher_min_classes(1, 1)
schedule1.set_teacher_min_classes(2, 1)
schedule1.set_teacher_max_classes(1, 5)
schedule1.set_teacher_max_classes(2, 4)
schedule1.display_schedule()
