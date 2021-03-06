# Copyright 2014-2019 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This file defines date handling logic.

Frequently we wish to use UTC dates, with a maximum of millisecond precision.
Using the methods defined in this file to create and convert dates ensures
equivalency when converting between timestamp and datetime formats.
"""
import datetime
import typing

import pytz

UTC = pytz.UTC
US_PACIFIC = pytz.timezone("US/Pacific")


def GetUtcMillisecondsNow() -> datetime.datetime:
  """Return the current date to millisecond precision.

  This method strips the microseconds returned value, allowing for equivalency
  checks before and after conversion to millisecond timestamps.

  Returns:
    A datetime instance.
  """
  d = datetime.datetime.utcnow()
  # Strip the microseconds. Don't round them to the nearest millisecond, as
  # this may cause rounding up beyond the range of valid microsecond values.
  return d.replace(microsecond=int(d.microsecond / 1000) * 1000)


<<<<<<< HEAD
<<<<<<< HEAD:labm8/py/labdate.py
<<<<<<< HEAD
<<<<<<< HEAD:labm8/py/labdate.py
def MillisecondsTimestamp(
  date: typing.Optional[datetime.datetime] = None,
) -> int:
=======
def MillisecondsTimestamp(date: typing.Optional[datetime.datetime] = None,
                         ) -> int:
>>>>>>> 49340dc00... Auto-format labm8 python files.:labm8/labdate.py
=======
def MillisecondsTimestamp(
  date: typing.Optional[datetime.datetime] = None,
) -> int:
>>>>>>> 4242aed2a... Automated code format.
=======
def MillisecondsTimestamp(date: typing.Optional[datetime.datetime] = None,
                         ) -> int:
>>>>>>> 49340dc00... Auto-format labm8 python files.:labm8/labdate.py
=======
def MillisecondsTimestamp(
  date: typing.Optional[datetime.datetime] = None,
) -> int:
>>>>>>> 4242aed2a... Automated code format.
  """Get the millisecond timestamp of a date.

  Args:
    date: A datetime instance. If not provided, GetUtcMillisecondsNow is used.

  Returns:
    The milliseconds since the epoch of this date.

  Raises:
    TypeError: If the argument is of incorrect type.
  """
  date = date or GetUtcMillisecondsNow()
  if not isinstance(date, datetime.datetime):
    raise TypeError("Date must be a datetime instance")
  return int(date.strftime("%s%f")[:-3])


<<<<<<< HEAD
<<<<<<< HEAD:labm8/py/labdate.py
<<<<<<< HEAD
<<<<<<< HEAD:labm8/py/labdate.py
def DatetimeFromMillisecondsTimestamp(
<<<<<<< HEAD:labm8/py/labdate.py
  timestamp: int = None,
) -> datetime.datetime:
=======
    timestamp: int = None) -> datetime.datetime:
>>>>>>> 303e215d5... Add default arg to labdate function.:lib/labm8/labdate.py
=======
def DatetimeFromMillisecondsTimestamp(timestamp: int = None,
                                     ) -> datetime.datetime:
>>>>>>> 49340dc00... Auto-format labm8 python files.:labm8/labdate.py
=======
def DatetimeFromMillisecondsTimestamp(
  timestamp: int = None,
) -> datetime.datetime:
>>>>>>> 4242aed2a... Automated code format.
=======
def DatetimeFromMillisecondsTimestamp(timestamp: int = None,
                                     ) -> datetime.datetime:
>>>>>>> 49340dc00... Auto-format labm8 python files.:labm8/labdate.py
=======
def DatetimeFromMillisecondsTimestamp(
  timestamp: int = None,
) -> datetime.datetime:
>>>>>>> 4242aed2a... Automated code format.
  """Get the date of a millisecond timestamp.

  Args:
    timestamp: Milliseconds since the epoch. If not provided, or if value is
      zero, the current time is used.

  Returns:
    A datetime instance.

  Raises:
    TypeError: If the argument is of incorrect type.
    ValueError: If the argument is not a positive integer.
  """
  if not (isinstance(timestamp, int) or timestamp is None):
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD:labm8/py/labdate.py
    raise TypeError("Timestamp must be an integer")
=======
    raise TypeError('Timestamp must be an integer')
>>>>>>> 303e215d5... Add default arg to labdate function.:lib/labm8/labdate.py
=======
    raise TypeError("Timestamp must be an integer")
>>>>>>> 4242aed2a... Automated code format.
=======
    raise TypeError("Timestamp must be an integer")
>>>>>>> 4242aed2a... Automated code format.
  if not timestamp:
    timestamp = MillisecondsTimestamp()
  if timestamp < 0:
    raise ValueError("Negative timestamp not allowed")
  return datetime.datetime.fromtimestamp(timestamp / 1000)
