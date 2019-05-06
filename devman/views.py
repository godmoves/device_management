from django.http import Http404
from django.shortcuts import render

from .models import Device, Sensor


def overview(request):
    device_list = Device.objects.order_by('-device_name')[::-1]
    device_num = len(device_list)
    device_running_num = len(Device.objects.filter(device_status='running'))
    device_stop_num = len(Device.objects.filter(device_status='stop'))
    device_repairing_num = len(Device.objects.filter(device_status='repairing'))
    device_error_num = len(Device.objects.filter(device_status='error'))
    side_bar = "overview"

    context = {'device_list': device_list,
               'device_num': device_num,
               'device_running_num': device_running_num,
               'device_stop_num': device_stop_num,
               'device_repairing_num': device_repairing_num,
               'device_error_num': device_error_num,
               'side_bar': side_bar}
    return render(request, 'devman/overview.html', context)


def detail(request, device_id):
    try:
        device = Device.objects.get(pk=device_id)
    except Device.DoesNotExist:
        raise Http404('Device does not exist')
    return render(request, 'devman/detail.html', {'device': device})


def repair(request, device_id):
    try:
        device = Device.objects.get(pk=device_id)
    except Device.DoesNotExist:
        raise Http404('Device does not exist')
    return render(request, 'devman/repair.html', {'device': device})


def error(request, device_id):
    try:
        device = Device.objects.get(pk=device_id)
    except Device.DoesNotExist:
        raise Http404('Device does not exist')
    return render(request, 'devman/error.html', {'device': device})


def plan(request):
    plan_total = 2000
    plan_complete = 1588
    plan_delay2 = 78
    plan_delay4 = 29
    plan_delay = plan_delay4 + plan_delay2
    plan_not_do = plan_total - plan_complete - plan_delay2 - plan_delay4

    device_num = len(Device.objects.all())
    device_running_num = len(Device.objects.filter(device_status='running'))
    device_stop_num = len(Device.objects.filter(device_status='stop'))
    device_repairing_num = len(Device.objects.filter(device_status='repairing'))
    device_error_num = len(Device.objects.filter(device_status='error'))

    context = {'device_num': device_num,
               'device_running_num': device_running_num,
               'device_stop_num': device_stop_num,
               'device_repairing_num': device_repairing_num,
               'device_error_num': device_error_num,
               'plan_total': plan_total,
               'plan_complete': plan_complete,
               'plan_delay2': plan_delay2,
               'plan_delay4': plan_delay4,
               'plan_delay': plan_delay,
               'plan_not_do': plan_not_do}
    return render(request, 'devman/plan.html', context)


def hist(request):
    return render(request, 'devman/hist.html', {})


def sensor(request, sensor_type, sensor_id):
    try:
        sensor = Sensor.objects.get(pk=sensor_id)
    except Sensor.DoesNotExist:
        raise Http404('Sensor does not exist')
    if sensor_type == 'hydraulic':
        return render(request, 'devman/sensor_hydraulic.html', {'sensor': sensor})
    elif sensor_type == 'bearing':
        return render(request, 'devman/sensor_bearing.html', {'sensor': sensor})
    elif sensor_type == 'trolley':
        return render(request, 'devman/sensor_trolley.html', {'sensor': sensor})
    elif sensor_type == 'gear':
        return render(request, 'devman/sensor_gear.html', {'sensor': sensor})
    else:
        raise Http404('Sensor type does not exist')

def device_error(request):
    device_list = Device.objects.filter(device_status='error')
    device_num = len(Device.objects.all())
    device_running_num = len(Device.objects.filter(device_status='running'))
    device_stop_num = len(Device.objects.filter(device_status='stop'))
    device_repairing_num = len(Device.objects.filter(device_status='repairing'))
    device_error_num = len(Device.objects.filter(device_status='error'))
    side_bar = "error"

    context = {'device_list': device_list,
               'device_num': device_num,
               'device_running_num': device_running_num,
               'device_stop_num': device_stop_num,
               'device_repairing_num': device_repairing_num,
               'device_error_num': device_error_num,
               'side_bar': side_bar}
    return render(request, 'devman/index.html', context)


def device_repairing(request):
    device_list = Device.objects.filter(device_status='repairing')
    device_num = len(Device.objects.all())
    device_running_num = len(Device.objects.filter(device_status='running'))
    device_stop_num = len(Device.objects.filter(device_status='stop'))
    device_repairing_num = len(Device.objects.filter(device_status='repairing'))
    device_error_num = len(Device.objects.filter(device_status='error'))
    side_bar = 'repairing'

    context = {'device_list': device_list,
               'device_num': device_num,
               'device_running_num': device_running_num,
               'device_stop_num': device_stop_num,
               'device_repairing_num': device_repairing_num,
               'device_error_num': device_error_num,
               'side_bar': side_bar}
    return render(request, 'devman/index.html', context)


def device_running(request):
    device_list = Device.objects.filter(device_status='running')
    device_num = len(Device.objects.all())
    device_running_num = len(Device.objects.filter(device_status='running'))
    device_stop_num = len(Device.objects.filter(device_status='stop'))
    device_repairing_num = len(Device.objects.filter(device_status='repairing'))
    device_error_num = len(Device.objects.filter(device_status='error'))
    side_bar = 'running'

    context = {'device_list': device_list,
               'device_num': device_num,
               'device_running_num': device_running_num,
               'device_stop_num': device_stop_num,
               'device_repairing_num': device_repairing_num,
               'device_error_num': device_error_num,
               'side_bar': side_bar}
    return render(request, 'devman/index.html', context)


def device_stop(request):
    device_list = Device.objects.filter(device_status='stop')
    device_num = len(Device.objects.all())
    device_running_num = len(Device.objects.filter(device_status='running'))
    device_stop_num = len(Device.objects.filter(device_status='stop'))
    device_repairing_num = len(Device.objects.filter(device_status='repairing'))
    device_error_num = len(Device.objects.filter(device_status='error'))
    side_bar = 'stop'

    context = {'device_list': device_list,
               'device_num': device_num,
               'device_running_num': device_running_num,
               'device_stop_num': device_stop_num,
               'device_repairing_num': device_repairing_num,
               'device_error_num': device_error_num,
               'side_bar': side_bar}
    return render(request, 'devman/index.html', context)
