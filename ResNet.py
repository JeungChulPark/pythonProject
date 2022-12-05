class Calc:
    def __init__(self, n1, n2):
        self.n1 = n1
        self.n2 = n2
        return print(self.n1, self.n2)

    def __call__(self, n1, n2):
        # self.n1 = n1
        # self.n2 = n2

        return print(self.n1 + self.n2)


s = Calc(1,2)

s(7,8) # call the __call__ function

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def get_name(self):
        print(f'Person : 제 이름은 {self.name}입니다.')

    def get_age(self):
        print(f'Person : 제 나이는 {self.age}세 입니다.')

class Student():
    def __init__(self, name, age, GPA):
        self.name = name
        self.age = age
        self.GPA = GPA

    def get_name(self):
        print(f'Student : 제 이름은 {self.name}입니다.')

    def get_age(self):
        print(f'Student : 제 나이는 {self.age}세 입니다.')

    def get_GPA(self):
        print(f'Student : 제 학점은 {self.GPA}입니다.')

class Student(Person):
    def __init__(self, name, age, GPA):
        super(Student, self).__init__(name, age)
        self.GPA = GPA

    def get_name(self):
        print(f'Student(Person) : 저는 대학생 {self.name}입니다.')

    def get_GPA(self):
        print(f'Student(Person) : 제 학점은 {self.GPA}입니다.')

student_a = Student('김땡떙', 27, 3.4)
student_a.get_name()
student_a.get_GPA()
student_a.get_age()

# arr = list(range(10))
# print(arr)
# print(arr[0:4])
# print(arr[0:7:3])
# print(arr[9:0:-1])
# print(arr[9:7:-1])

# class A:
#     def __getitem__(self, item):
#         print(repr(item))
#
# a = A()
# a[1]
# a[1:2]
# a[1:2:3]
# a[1,2,3]
# a[1,2,3:4]
# a[1,2,3:4,...]

import collections

Card = collections.namedtuple('Card', ['rank', 'suit'])

class FrenchDeck:
    ranks = [str(n) for n in range(2, 11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()

    def __init__(self):
        print(self.ranks)
        print(self.suits)
        self._cards = [Card(rank, suit) for suit in self.suits for rank in self.ranks]
        print(self._cards)

    def __len__(self):
        return len(self._cards)

    def __getitem__(self, position):
        return self._cards[position]

deck = FrenchDeck()
print(len(deck))
print(deck[0])
print(deck[-1])
print(deck[len(deck)-1])

from random import choice
print(choice(deck))
print(choice(deck))
print(choice(deck))

suit_values = dict(spades=3, hearts=2, diamonds=1, clubs=0)

def spades_high(card):
    rank_value = FrenchDeck.ranks.index(card.rank)
    return rank_value * len(suit_values) + suit_values[card.suit]

for card in sorted(deck, key=spades_high):
    print(card)

print(deck[:3])

Card('Q', 'hearts') in deck
Card('7', 'beasts') in deck