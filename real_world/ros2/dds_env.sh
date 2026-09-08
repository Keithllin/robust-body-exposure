#!/usr/bin/env bash
# ROS 2 FastDDS env for RCHI <-> Stretch. Source after Humble.
# Sets ROS_DOMAIN_ID=12, ROS_IP to the address that reaches the peer,
# and a unicast peer XML (campus WiFi drops multicast).
#
# Override peers if DHCP changes:
#   export ROBE_STRETCH_IP=... ROBE_WORKSTATION_IP=...
_robe_had_u=0
case "$-" in
  *u*) _robe_had_u=1 ;;
esac
set +u

: "${ROBE_STRETCH_IP:?Set ROBE_STRETCH_IP to the Stretch peer address}"
: "${ROBE_WORKSTATION_IP:?Set ROBE_WORKSTATION_IP to the workstation address}"

_robe_src_ip_to() {
  ip -4 route get "$1" 2>/dev/null \
    | awk '{for (i = 1; i <= NF; i++) if ($i == "src") { print $(i + 1); exit }}'
}

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-12}"
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

_peer="${ROBE_STRETCH_IP}"
_me="$(_robe_src_ip_to "${ROBE_STRETCH_IP}")"
if [[ -n "${_me}" && "${_me}" == "${ROBE_STRETCH_IP}" ]]; then
  _peer="${ROBE_WORKSTATION_IP}"
  _me="$(_robe_src_ip_to "${ROBE_WORKSTATION_IP}")"
fi
if [[ -z "${_me}" ]]; then
  _me="$(_robe_src_ip_to 1.1.1.1)"
fi
if [[ -n "${_me}" ]]; then
  export ROS_IP="${_me}"
fi
unset ROS_HOSTNAME

# Unicast XML is for RCHI → Stretch on campus WiFi. The same default
# participant profile on extra Stretch processes (preview_origin) does
# not receive the local D435i. Stretch only needs ROS_IP.
_host="$(hostname 2>/dev/null || true)"
if [[ "${_host}" == stretch* || -n "${HELLO_FLEET_ID:-}" ]]; then
  unset FASTRTPS_DEFAULT_PROFILES_FILE
else
  _xml_dir="${HOME}/.ros"
  mkdir -p "${_xml_dir}"
  _xml="${_xml_dir}/robe_fastrtps_unicast.xml"
  cat > "${_xml}" <<EOF
<?xml version="1.0" encoding="UTF-8" ?>
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
    <participant profile_name="robe_unicast" is_default_profile="true">
        <rtps>
            <builtin>
                <initialPeersList>
                    <locator>
                        <udpv4>
                            <address>${ROBE_STRETCH_IP}</address>
                        </udpv4>
                    </locator>
                    <locator>
                        <udpv4>
                            <address>${ROBE_WORKSTATION_IP}</address>
                        </udpv4>
                    </locator>
                </initialPeersList>
            </builtin>
        </rtps>
    </participant>
</profiles>
EOF
  export FASTRTPS_DEFAULT_PROFILES_FILE="${_xml}"
fi
unset _peer _me _xml _xml_dir _host
if [[ "${_robe_had_u}" == 1 ]]; then
  set -u
fi
unset _robe_had_u
