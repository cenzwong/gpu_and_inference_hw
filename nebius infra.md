```terraform
variable "user_data" {
  default = <<EOT
users:
 - name: cenzwong
   sudo: ALL=(ALL) NOPASSWD:ALL
   shell: /bin/bash
   ssh_authorized_keys:
    - ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAFsVmiT/M2/50jq0dzd6yWWzumDGy4X94snrsQNGYY7 cenzwong@192.168.1.102
  EOT
}

resource "nebius_compute_v1_disk" "boot_disk" {
  name = "cenz-wong-instance-8-boot-disk"
  parent_id = "project-e00k1qhxpr00c8m6d123nq"
  type = "NETWORK_SSD"
  block_size_bytes = 4096
  size_bytes = 1374389534720
  source_image_family = {
    image_family = "ubuntu24.04-cuda13.0"
  }
}

resource "nebius_compute_v1_instance" "vm" {
  name = "cenz-wong-instance-8"
  parent_id = "project-e00k1qhxpr00c8m6d123nq"
  stopped = false
  resources = {
    platform = "gpu-l40s-d"
    preset = "1gpu-16vcpu-96gb"
  }
  boot_disk = {
    existing_disk = {
      id = nebius_compute_v1_disk.boot_disk.id
    }
    attach_mode = "READ_WRITE"
    device_id = "boot-disk"
  }
  network_interfaces = [
    {
      name = "eth0"
      subnet_id = "vpcsubnet-e00bq02c5h869qycv6"
      ip_address = {}
      public_ip_address = {}
    }
  ]
  cloud_init_user_data = var.user_data
  reservation_policy = {
    policy = "AUTO"
  }
}
```

```sh
export VM_BOOT_DISK_ID=$(nebius compute disk create \
  --name cenz-wong-instance-8-boot-disk \
  --parent-id project- \
  --type network_ssd \
  --block-size-bytes 4096 \
  --size-bytes 1374389534720 \
  --source-image-family-image-family ubuntu24.04-cuda13.0 --format json | jq -r '.metadata.id')

nebius compute instance create \
  --name cenz-wong-instance-8 \
  --parent-id project- \
  --stopped false \
  --resources-platform gpu-l40s-d \
  --resources-preset 1gpu-16vcpu-96gb \
  --boot-disk-existing-disk-id $VM_BOOT_DISK_ID \
  --boot-disk-attach-mode read_write \
  --boot-disk-device-id boot-disk \
  --network-interfaces '[{"name":"eth0","ip_address":{},"subnet_id":"vpcsubnet-e00bq02c5h869qycv6","public_ip_address":{}}]' \
  --cloud-init-user-data $'users:
 - name: cenzwong
   sudo: ALL=(ALL) NOPASSWD:ALL
   shell: /bin/bash
   ssh_authorized_keys:
    - ssh-ed25519 AAAA cenzwong@192.168.1.102' \
  --reservation-policy-policy auto
```