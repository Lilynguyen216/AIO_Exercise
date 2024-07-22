class Student:
    def __init__(self, name, yob, grade):
        self._name = name
        self._yob = yob
        self._grade = grade

    def describe(self):
        return f"Student: {self._name}, Year of Birth: {self._yob}, Grade: {self._grade}"


class Doctor:
    def __init__(self, name, yob, specialist):
        self._name = name
        self._yob = yob
        self._specialist = specialist

    def describe(self):
        return f"Doctor: {self._name}, Year of Birth: {self._yob}, Specialist: {self._specialist}"


class Teacher:
    def __init__(self, name, yob, subject):
        self._name = name
        self._yob = yob
        self._subject = subject

    def describe(self):
        return f"Teacher: {self._name}, Year of Birth: {self._yob}, Subject: {self._subject}"


class Ward:
    def __init__(self, name):
        self.name = name
        self.people = []

    def add_person(self, person):
        for _person in self.people:
          if person._name == _person._name and person._yob == _person._yob:
            return
        self.people.append(person)

    def describe(self):
        print(f"Ward: {self.name}")
        for person in self.people:
            print(person.describe())

    def count_doctor(self):
        return sum(1 for person in self.people if isinstance(person, Doctor))

    def sort_age(self):
        self.people.sort(key=lambda person: person._yob)

    def compute_average(self):
        teachers = [
            person for person in self.people if isinstance(person, Teacher)]
        if teachers:
            return sum(teacher._yob for teacher in teachers) / len(teachers)
        return None


# Tạo đối tượng Ward và thêm người vào
ward = Ward("Ward 1")
ward.add_person(Teacher("A", 1975, "Science"))
ward.add_person(Doctor("B", 1965, "Cardiolog"))
ward.add_person(Doctor("C", 1970, "Neurology"))

# Tính trung bình năm sinh của các giáo viên
average_yob = ward.compute_average()
if average_yob is not None:
    print("Average Year of Birth for Teachers:", average_yob)
else:
    print("No teachers in the ward.")
