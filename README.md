**Status:** Archive (code is provided as-is, no updates expected)

# Device Management

## Set up environment

- Install [Python3](https://www.python.org/downloads/)
- Install [Django](https://www.djangoproject.com/download/)
- Install [Bootstrap3](https://getbootstrap.com/docs/3.4/getting-started/)

## Start the project

Open the terminal and type:
```
python manage.py runserver
```

## File structure

### Models

All objects (devices & sensors) and their properties are set up in `devman/models.py`

### URLs

The pages' links are defined in `devman/urls.py`

### Views

Data processing and template picking, see `devman/views.py`

### Templates

All HTML pages are defined in `devman/templates/*.html`

### Administrator

Pages related to administrator are saved in `templates`

### Libraries & resources

Third-party files and resources are stored in `static`

### Database

We use SQLite. All example data in `db.sqlite3`

## Credit

**Author:** ManKit Pong  
&copy; All right reserved.

