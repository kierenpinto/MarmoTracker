#!/usr/bin/python

import sqlite3

class SQLStore:
    def __init__(self,location):
        self.conn = sqlite3.connect(location)
        self.cursor = self.conn.cursor()

    def createTable(self):
        c.execute('''CREATE TABLE data_record
             (EntryId integer, timestamp datetime, NovelMarmoset integer, qty real, price real)''')

    def add(self):
        self.c.execute()


class PythonStore:
    def __init__(self):
