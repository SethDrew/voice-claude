-- voice-router hotkey — double-tap Fn to activate
local voiceRouter = {}
voiceRouter.lastFnTime = 0
voiceRouter.doubleTapInterval = 0.4  -- seconds

voiceRouter.tap = hs.eventtap.new({hs.eventtap.event.types.flagsChanged}, function(event)
    local flags = event:getRawEventData().CGEventData.flags
    -- Fn key = 0x800000
    local fnPressed = (flags & 0x800000) ~= 0

    if fnPressed then
        local now = hs.timer.secondsSinceEpoch()
        local delta = now - voiceRouter.lastFnTime
        voiceRouter.lastFnTime = now

        if delta < voiceRouter.doubleTapInterval then
            -- Double-tap detected
            voiceRouter.lastFnTime = 0  -- reset to prevent triple-tap

            hs.alert.show("🎤 Listening...", nil, nil, 1.5)

            local env = {
                PATH = os.getenv("HOME") .. "/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
                HOME = os.getenv("HOME"),
            }

            local task = hs.task.new(
                os.getenv("HOME") .. "/.local/bin/voice-route",
                function(exitCode, stdout, stderr)
                    if exitCode == 0 and stdout and #stdout > 0 then
                        hs.alert.show("✓ " .. stdout:sub(1, 80), nil, nil, 2)
                    else
                        hs.alert.show("✗ Voice route failed", nil, nil, 2)
                    end
                end,
                {"--hotkey"}
            )
            task:setEnvironment(env)
            task:start()
        end
    end

    return false
end)

voiceRouter.tap:start()
