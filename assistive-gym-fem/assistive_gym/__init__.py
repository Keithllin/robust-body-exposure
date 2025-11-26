from gym.envs.registration import register

tasks = ['Bedding']
robots = ['Stretch']

for task in tasks:
    for robot in robots:
        register(
            id='%s%s-v1' % (task, robot),
            entry_point='assistive_gym.envs:%s%sEnv' % (task, robot),
            max_episode_steps=200,
        )

register(
    id='RobeReversible-v1',
    entry_point='assistive_gym.envs:RobeReversibleEnv',
    max_episode_steps=200,
)
register(
    id='RobeReversible2-v1',
    entry_point='assistive_gym.envs:RobeReversibleEnv2',
    max_episode_steps=200,
)


