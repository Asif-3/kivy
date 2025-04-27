[app]

# (str) Title of your application
title = Movie Recommendation App

# (str) Package name
package.name = asif

# (str) Package domain (needed for android/ios packaging)
package.domain = org.test

# (str) Source code where the main.py lives
source.dir = .

# (str) Main .py file to start your app
source.main = main.py

# (list) Source files to include
source.include_exts = py,png,jpg,kv,atlas,csv

# (list) Source files to exclude
#source.exclude_exts = spec

# (list) List of directory to exclude (let empty to not exclude anything)
#source.exclude_dirs = tests, bin, venv

# (str) Application version
version = 0.1

# (list) Application requirements
requirements = python3,kivy,pillow

# (list) Supported orientations
orientation = portrait

# (bool) Indicate if the application should be fullscreen or not
fullscreen = 0

# (int) Android API level to target
android.api = 31

# (int) Minimum API your APK will support
android.minapi = 21

# (int) Android NDK API to use
android.ndk_api = 21

# (list) The Android architectures to build for
android.archs = arm64-v8a,armeabi-v7a

# (bool) Enable AndroidX support
android.enable_androidx = True

# (bool) Automatically accept SDK licenses
android.accept_sdk_license = True

# (bool) Copy library instead of making a libpymodules.so
android.copy_libs = 1

# (bool) Enable backup
android.allow_backup = True

#
# iOS specific
#

# (bool) Whether or not to sign the code
ios.codesign.allowed = false
