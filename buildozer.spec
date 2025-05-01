[app]

# (str) Title of your application
title = Movie Recommendation System

# (str) Package name (no space after equals sign)
package.name = asif

# (str) Package domain (needed for android/ios packaging)
package.domain = org.test

# (str) Source code where the main.py live
source.dir = .

# (list) Source files to include (leave empty to include all the files)
source.include_exts = py,png,jpg,kv,atlas,csv

# (list) List of inclusions using pattern matching - fixed the pattern format
source.include_patterns = assets/*,images/*.png,images/*.csv

# (str) Application versioning
version = 0.1

# (list) Application requirements - use specific versions for stability
requirements = python3,kivy==2.1.0,kivymd==0.104.2,pillow==8.3.1

# (str) Supported orientations
orientation = portrait

# Android specific
android.permissions = android.permission.INTERNET
android.api = 31
android.minapi = 21
android.sdk = 31
android.ndk = 23b
android.ndk_api = 21
android.archs = arm64-v8a, armeabi-v7a
android.allow_backup = True

# Enable automatic SDK license acceptance for CI/CD
android.accept_sdk_license = True

# Skip updates to speed up builds
android.skip_update = True

# Python for Android (p4a) settings
p4a.bootstrap = sdl2

[buildozer]
log_level = 2
warn_on_root = 1
